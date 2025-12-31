#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bulk loader for Job or Resume JSONL payloads (pick one via --kind).
- JobPost JSONL: writes Company, JobPost, Location, Occupation, REPRESENTS_OCCUPATION, Skills, Chunks
- Resume  JSONL: writes Candidate, HAS_SKILL, HELD_OCCUPATION, Chunks
"""

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from tqdm import tqdm

# Minimum confidence required to keep a skill
SKILL_CONF_THRESHOLD = 0.9

# ---------- DDL ----------
DDL_BASE = [
    "CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:JobPost) REQUIRE j.job_id IS UNIQUE",
    "CREATE CONSTRAINT cand_id_unique IF NOT EXISTS FOR (c:Candidate) REQUIRE c.cand_id IS UNIQUE",
    "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
    "CREATE INDEX job_dates IF NOT EXISTS FOR (j:JobPost) ON (j.posted_on, j.expiry_date)",
    "CREATE INDEX job_type IF NOT EXISTS FOR (j:JobPost) ON (j.employment_type)",
    "CREATE CONSTRAINT location_key_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.loc_key IS UNIQUE",
    "CREATE INDEX location_name IF NOT EXISTS FOR (l:Location) ON (l.name)",
]

DDL_VECTOR_NEW = """
CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
FOR (ch:Chunk) ON (ch.vector)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }
"""

DDL_VECTOR_LEGACY = """
CREATE INDEX chunk_vector_index IF NOT EXISTS
FOR (ch:Chunk) ON (ch.vector)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } }
"""

# ---------- Cypher (bulk UNWIND) ----------
CYPHER_BATCH_JOB = """
UNWIND $batch AS row
MERGE (c:Company {name: row.company})

// Job
MERGE (j:JobPost {job_id: row.job_id})
ON CREATE SET j.created_at = datetime()
SET j.title           = row.title,
    j.url             = row.url,
    j.employment_type = row.employment_type,
    j.salary_min      = row.salary_min,
    j.salary_max      = row.salary_max,
    j.currency        = row.currency,
    j.period          = row.period,
    j.posted_on       = row.posted_on,
    j.expiry_date     = row.expiry_date,
    j.status          = row.status,
    j.updated_at      = datetime()
MERGE (c)-[:POSTED]->(j)

// Location (conditional)
FOREACH (_ IN CASE WHEN row.loc_key IS NULL OR row.loc_key = '' THEN [] ELSE [1] END |
  MERGE (loc:Location {loc_key: row.loc_key})
  FOREACH (_ IN CASE WHEN row.location IS NULL OR row.location = '' THEN [] ELSE [1] END | SET loc.name = row.location)
  MERGE (j)-[:LOCATED_IN]->(loc)
)

// Occupations (REPRESENTS_OCCUPATION)
// FOREACH (o IN COALESCE(row.occupations, []) |
//   MERGE (occ:Occupation {name: o})
//   MERGE (j)-[:REPRESENTS_OCCUPATION]->(occ)
// )

// Skills
FOREACH (sk IN COALESCE(row.skills, []) |
  MERGE (s:Skill {name: sk.name})
  MERGE (j)-[r:REQUIRES_SKILL]->(s)
  FOREACH (_ IN CASE WHEN sk.confidence IS NULL THEN [] ELSE [1] END | SET r.confidence = sk.confidence)
)

// Replace chunks for this job
WITH j, row
OPTIONAL MATCH (j)-[:HAS_CHUNK]->(oldc:Chunk)
DETACH DELETE oldc
WITH j, row
FOREACH (ch IN COALESCE(row.chunks, []) |
  MERGE (cn:Chunk {chunk_id: ch.chunk_id})
  SET cn.text = ch.text,
      cn.vector = CASE WHEN ch.vector IS NULL THEN cn.vector ELSE ch.vector END
  MERGE (j)-[:HAS_CHUNK]->(cn)
)
"""

CYPHER_BATCH_RESUME = """
UNWIND $batch AS row
// Candidate
MERGE (c:Candidate {cand_id: row.cand_id})
ON CREATE SET c.created_at = datetime()
SET c.name = row.name,
    c.updated_at = datetime()

// HAS_SKILL
FOREACH (sk IN COALESCE(row.skills, []) |
  MERGE (s:Skill {name: sk.name})
  MERGE (c)-[h:HAS_SKILL]->(s)
  FOREACH (_ IN CASE WHEN sk.confidence IS NULL THEN [] ELSE [1] END | SET h.confidence = sk.confidence)
)

// HELD_OCCUPATION
// FOREACH (o IN COALESCE(row.occupations, []) |
//   MERGE (occ:Occupation {name: o})
//   MERGE (c)-[:HELD_OCCUPATION {title: o}]->(occ)
// )

// Replace chunks for this candidate
// WITH c, row
// OPTIONAL MATCH (c)-[:HAS_CHUNK]->(oldc:Chunk)
// DETACH DELETE oldc
// WITH c, row
// FOREACH (ch IN COALESCE(row.chunks, []) |
//   MERGE (cn:Chunk {chunk_id: ch.chunk_id})
//   SET cn.text = ch.text,
//       cn.vector = CASE WHEN ch.vector IS NULL THEN cn.vector ELSE ch.vector END
//   MERGE (c)-[:HAS_CHUNK]->(cn)
// )
"""

# ---------- Helpers ----------
def _norm_loc_key(location: Optional[str]) -> str:
    if location and location.strip():
        return location.strip().lower()
    return ""

def ensure_schema(driver):
    with driver.session() as s:
        for stmt in DDL_BASE:
            s.run(stmt)
        try:
            s.run(DDL_VECTOR_NEW)
        except ClientError:
            try:
                s.run(DDL_VECTOR_LEGACY)
            except ClientError:
                pass

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

# ---------- Mappers ----------
def map_job_payload(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "graph" not in p or "nodes" not in p["graph"]:
        return None
    nodes = p["graph"]["nodes"]
    triples = p["graph"].get("triples", [])

    job_key = next((k for k, v in nodes.items() if v.get("type") == "JobPost"), None)
    if not job_key:
        return None
    j = nodes[job_key]
    job_id = j.get("job_id")
    if not job_id:
        return None

    # company
    company = "Unknown"
    for v in nodes.values():
        if v.get("type") == "Company":
            company = v.get("name") or "Unknown"
            break

    # location
    location = j.get("location")
    if not location:
        for v in nodes.values():
            if v.get("type") == "Location":
                location = v.get("name") or v.get("location") or v.get("value") or v.get("raw")
                if location:
                    break
    loc_key = _norm_loc_key(location)

    # skills (threshold)
    conf_map = {}
    for h, r, t, *rest in triples:
        if r == "REQUIRES_SKILL":
            conf = (rest[0] or {}).get("confidence") if rest else None
            try:
                conf = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf = None
            conf_map[t] = conf

    skills: List[Dict[str, Any]] = []
    for k, v in nodes.items():
        if v.get("type") == "Skill":
            conf = conf_map.get(k)
            if conf is not None and conf >= SKILL_CONF_THRESHOLD:
                skills.append({"name": v.get("name"), "confidence": conf})

    # occupations from REPRESENTS_OCCUPATION
    # occ_ids = set()
    # for h, r, t, *rest in triples:
    #     if r == "REPRESENTS_OCCUPATION" and t in nodes and nodes[t].get("type") == "Occupation":
    #         occ_ids.add(t)
    # occupations: List[str] = []
    # for oid in occ_ids:
    #     nm = (nodes[oid].get("name") or "").strip()
    #     if nm:
    #         occupations.append(nm)

    chunks = p.get("chunks", [])
    return {
        "job_id": job_id,
        "title": j.get("title"),
        "url": j.get("url"),
        "company": company,
        "employment_type": j.get("employment_type"),
        "salary_min": j.get("salary_min"),
        "salary_max": j.get("salary_max"),
        "currency": j.get("currency"),
        "period": j.get("period"),
        "posted_on": j.get("posted_on"),
        "expiry_date": j.get("expiry_date"),
        "status": j.get("status"),
        "location": (location or "").strip(),
        "loc_key": loc_key,
        "skills": skills,
        # "occupations": occupations,
        "chunks": chunks,
    }

def map_resume_payload(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "graph" not in p or "nodes" not in p["graph"]:
        return None
    nodes = p["graph"]["nodes"]
    triples = p["graph"].get("triples", [])

    cand_key = next((k for k, v in nodes.items() if v.get("type") == "Candidate"), None)
    if not cand_key:
        return None
    c = nodes[cand_key]
    cand_id = c.get("cand_id")
    if not cand_id:
        return None

    name = c.get("name")

    # skills (threshold)
    conf_map = {}
    for h, r, t, *rest in triples:
        if r == "HAS_SKILL":
            conf = (rest[0] or {}).get("confidence") if rest else None
            try:
                conf = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf = None
            conf_map[t] = conf

    skills: List[Dict[str, Any]] = []
    for k, v in nodes.items():
        if v.get("type") == "Skill":
            conf = conf_map.get(k)
            if conf is not None and conf >= SKILL_CONF_THRESHOLD:
                skills.append({"name": v.get("name"), "confidence": conf})

    # occupations via HELD_OCCUPATION
    # occ_ids = set()
    # for h, r, t, *rest in triples:
    #     if r == "HELD_OCCUPATION" and t in nodes and nodes[t].get("type") == "Occupation":
    #         occ_ids.add(t)
    # occupations: List[str] = []
    # for oid in occ_ids:
    #     nm = (nodes[oid].get("name") or "").strip()
    #     if nm:
    #         occupations.append(nm)

    chunks = p.get("chunks", [])
    return {
        "cand_id": cand_id,
        "name": name,
        "skills": skills,
        # "occupations": occupations,
        "chunks": chunks,
    }

# ---------- Writer ----------
def write_batches(driver, rows: Iterable[Dict[str, Any]], kind: str, batch_size: int = 500):
    cypher = CYPHER_BATCH_JOB if kind == "job" else CYPHER_BATCH_RESUME
    batch: List[Dict[str, Any]] = []
    with driver.session() as s:
        for r in rows:
            if r is None:
                continue
            batch.append(r)
            if len(batch) >= batch_size:
                s.run(cypher, batch=batch)
                batch.clear()
        if batch:
            s.run(cypher, batch=batch)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Bulk load JSONL payloads into Neo4j (use --kind job|resume).")
    ap.add_argument("--jsonl", required=True, help="Path to JSONL from extract step")
    ap.add_argument("--kind", required=True, choices=["job", "resume"], help="Payload type in the JSONL")
    ap.add_argument("--uri", required=True, help="bolt://localhost:7687 or neo4j://host:7687")
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--skip-schema", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of records to load")
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    if not args.skip_schema:
        ensure_schema(driver)

    import itertools
    source_iter = read_jsonl(args.jsonl)
    if args.limit:
        source_iter = itertools.islice(source_iter, args.limit)

    if args.kind == "job":
        rows_iter = (map_job_payload(p) for p in tqdm(source_iter, desc="Reading job JSONL"))
    else:
        rows_iter = (map_resume_payload(p) for p in tqdm(source_iter, desc="Reading resume JSONL"))

    write_batches(driver, rows_iter, kind=args.kind, batch_size=args.batch_size)
    driver.close()
    print("✅ Bulk write complete.")

if __name__ == "__main__":
    main()