#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

from neo4j import Driver, GraphDatabase

# Update these defaults or override with environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")

_driver: Optional[Driver] = None
_driver_config: Dict[str, str] | None = None


def _resolve_config(uri: Optional[str], user: Optional[str], password: Optional[str]) -> Dict[str, str]:
    cfg = {
        "uri": uri or NEO4J_URI,
        "user": user or NEO4J_USER,
        "password": password or NEO4J_PASSWORD,
    }
    if not all(cfg.values()):
        raise ValueError("Neo4j URI, user, and password must be provided.")
    return cfg


def init_driver(uri: Optional[str] = None,
                user: Optional[str] = None,
                password: Optional[str] = None) -> Driver:
    """
    Force creation of a driver with the provided credentials.
    """
    global _driver, _driver_config
    cfg = _resolve_config(uri, user, password)
    if _driver is not None:
        _driver.close()
    _driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    _driver_config = cfg
    return _driver


def get_driver(uri: Optional[str] = None,
               user: Optional[str] = None,
               password: Optional[str] = None) -> Driver:
    """
    Return a cached driver. If credentials change, recreate it.
    """
    global _driver, _driver_config
    if uri is None and user is None and password is None:
        if _driver is None:
            return init_driver()
        return _driver
    cfg = _resolve_config(uri, user, password)
    if _driver is None or _driver_config != cfg:
        return init_driver(cfg["uri"], cfg["user"], cfg["password"])
    return _driver


def close_driver():
    global _driver, _driver_config
    if _driver is not None:
        _driver.close()
    _driver = None
    _driver_config = None


# --- Cypher snippets (cosine similarity using pure Cypher) ---

COSINE_SIM_EXPR = """
CASE
    WHEN {left} IS NOT NULL AND {right} IS NOT NULL THEN
        reduce(dot = 0.0, i IN range(0, size({left})-1) |
            dot + ({left}[i] * {right}[i])
        ) /
        (
            sqrt(reduce(sa = 0.0, i IN range(0, size({left})-1) |
                sa + ({left}[i] * {left}[i])
            )) *
            sqrt(reduce(sb = 0.0, i IN range(0, size({right})-1) |
                sb + ({right}[i] * {right}[i])
            ))
        )
    ELSE 0.0
END
"""


# --- Top jobs for candidate ---

TOP_JOBS_QUERY = f"""
// Set candidate and weights via params
WITH $candId AS candId,
     $alpha AS alpha,
     $beta  AS beta,
     $gamma AS gamma

MATCH (c:Candidate {{cand_id: candId}})-[:HAS_SKILL]->(s:Skill)
MATCH (s)<-[:REQUIRES_SKILL]-(j:JobPost)
WITH c, j, COLLECT(DISTINCT s) AS sharedSkills, alpha, beta, gamma

MATCH (j)-[:REQUIRES_SKILL]->(js:Skill)
WITH c, j, sharedSkills, COLLECT(DISTINCT js) AS jobSkills,
     alpha, beta, gamma

WITH c, j,
     size(sharedSkills) AS sharedCount,
     size(jobSkills)    AS jobSkillCount,
     alpha, beta, gamma
WHERE jobSkillCount > 0

WITH c, j, sharedCount, jobSkillCount,
     1.0 * sharedCount / jobSkillCount AS coverage,
     c.rasch_ability    AS theta_c,
     j.rasch_difficulty AS beta_j,
     alpha, beta, gamma

WITH c, j, sharedCount, jobSkillCount, coverage, alpha, beta, gamma,
     CASE
         WHEN theta_c IS NULL OR beta_j IS NULL
             THEN coverage
         ELSE 1.0 / (1 + exp(-(theta_c - beta_j)))
     END AS rasch_match

WITH c, j, sharedCount, jobSkillCount, coverage, rasch_match,
     alpha, beta, gamma,
     {COSINE_SIM_EXPR.format(left="c.transr", right="j.transr")} AS transr_sim

WITH c, j, sharedCount, jobSkillCount, coverage, rasch_match, transr_sim,
     (alpha * coverage) +
     (beta  * rasch_match) +
     (gamma * transr_sim) AS final_score

MATCH (j)<-[:POSTED]-(comp:Company)

RETURN j.job_id       AS job_id,
       j.title        AS jobTitle,
       comp.name      AS company,
       sharedCount,
       jobSkillCount,
       round(coverage,    3) AS coverage,
       round(rasch_match, 3) AS rasch_match,
       round(transr_sim,  3) AS transr_sim,
       round(final_score, 3) AS final_score
ORDER BY final_score DESC, coverage DESC, sharedCount DESC
LIMIT $k
"""


# --- Top candidates for job ---

TOP_CANDIDATES_QUERY = f"""
WITH $jobId AS jobId,
     $alpha AS alpha,
     $beta  AS beta,
     $gamma AS gamma

MATCH (j:JobPost {{job_id: jobId}})-[:REQUIRES_SKILL]->(s:Skill)
MATCH (s)<-[:HAS_SKILL]-(c:Candidate)
WITH j, c, COLLECT(DISTINCT s) AS sharedSkills, alpha, beta, gamma

MATCH (c)-[:HAS_SKILL]->(cs:Skill)
WITH j, c, sharedSkills, COLLECT(DISTINCT cs) AS candSkills,
     alpha, beta, gamma

WITH j, c,
     size(sharedSkills) AS sharedCount,
     size(candSkills)   AS candSkillCount,
     alpha, beta, gamma
WHERE candSkillCount > 0

WITH j, c, sharedCount, candSkillCount,
     1.0 * sharedCount / candSkillCount AS coverage,
     c.rasch_ability    AS theta_c,
     j.rasch_difficulty AS beta_j,
     alpha, beta, gamma

WITH j, c, sharedCount, candSkillCount, coverage, alpha, beta, gamma,
     CASE
         WHEN theta_c IS NULL OR beta_j IS NULL
             THEN coverage
         ELSE 1.0 / (1 + exp(-(theta_c - beta_j)))
     END AS rasch_match

WITH j, c, sharedCount, candSkillCount, coverage, rasch_match,
     alpha, beta, gamma,
     {COSINE_SIM_EXPR.format(left="c.transr", right="j.transr")} AS transr_sim

WITH j, c, sharedCount, candSkillCount, coverage, rasch_match, transr_sim,
     (alpha * coverage) +
     (beta  * rasch_match) +
     (gamma * transr_sim) AS final_score

MATCH (j)<-[:POSTED]-(comp:Company)

RETURN j.job_id       AS job_id,
       j.title        AS jobTitle,
       comp.name      AS company,
       c.cand_id      AS cand_id,
       c.name         AS candidateName,
       sharedCount,
       candSkillCount,
       round(coverage,    3) AS coverage,
       round(rasch_match, 3) AS rasch_match,
       round(transr_sim,  3) AS transr_sim,
       round(final_score, 3) AS final_score
ORDER BY final_score DESC, sharedCount DESC, coverage DESC
LIMIT $k
"""


def rank_jobs_for_candidate(cand_id: str, k: int = 10,
                            alpha: float = 1.0,
                            beta: float = 0.2,
                            gamma: float = 0.1,
                            driver: Optional[Driver] = None):
    drv = driver or get_driver()
    with drv.session() as session:
        result = session.run(
            TOP_JOBS_QUERY,
            candId=cand_id,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        return [r.data() for r in result]


def rank_candidates_for_job(job_id: str, k: int = 10,
                            alpha: float = 1.0,
                            beta: float = 0.2,
                            gamma: float = 0.1,
                            driver: Optional[Driver] = None):
    drv = driver or get_driver()
    with drv.session() as session:
        result = session.run(
            TOP_CANDIDATES_QUERY,
            jobId=job_id,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        return [r.data() for r in result]


def list_candidates(limit: int = 50, driver: Optional[Driver] = None):
    query = """
    MATCH (c:Candidate)
    WHERE c.transr IS NOT NULL
    RETURN c.cand_id AS cand_id, c.name AS name
    LIMIT $limit
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, limit=limit)
        return [r.data() for r in res]


def list_jobs(limit: int = 50, driver: Optional[Driver] = None):
    query = """
    MATCH (j:JobPost)<-[:POSTED]-(comp:Company)
    WHERE j.transr IS NOT NULL
    RETURN j.job_id AS job_id, j.title AS title, comp.name AS company
    LIMIT $limit
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, limit=limit)
        return [r.data() for r in res]


def search_candidates_by_name(name_query: str, limit: int = 20, driver: Optional[Driver] = None):
    if not name_query:
        return []
    query = """
    MATCH (c:Candidate)
    WHERE toLower(c.name) CONTAINS toLower($query)
    RETURN c.cand_id AS cand_id, c.name AS name
    ORDER BY c.name ASC
    LIMIT $limit
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, {"query": name_query, "limit": limit})
        return [r.data() for r in res]


def search_jobs_by_title(title_query: str, limit: int = 20, driver: Optional[Driver] = None):
    if not title_query:
        return []
    query = """
    MATCH (j:JobPost)<-[:POSTED]-(comp:Company)
    WHERE toLower(j.title) CONTAINS toLower($query)
    RETURN j.job_id AS job_id, j.title AS title, comp.name AS company
    ORDER BY j.title ASC
    LIMIT $limit
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, {"query": title_query, "limit": limit})
        return [r.data() for r in res]


def get_candidate_skills(cand_id: str, driver: Optional[Driver] = None) -> List[str]:
    query = """
    MATCH (c:Candidate {cand_id: $cand_id})-[:HAS_SKILL]->(s:Skill)
    RETURN DISTINCT s.name AS skill
    ORDER BY skill ASC
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, cand_id=cand_id)
        return [r["skill"] for r in res if r.get("skill")]


def get_job_skills(job_id: str, driver: Optional[Driver] = None) -> List[str]:
    query = """
    MATCH (j:JobPost {job_id: $job_id})-[:REQUIRES_SKILL]->(s:Skill)
    RETURN DISTINCT s.name AS skill
    ORDER BY skill ASC
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, job_id=job_id)
        return [r["skill"] for r in res if r.get("skill")]


def suggest_skills_for_candidate(cand_id: str,
                                 job_ids: Iterable[str],
                                 limit: int = 15,
                                 driver: Optional[Driver] = None):
    job_list = [jid for jid in job_ids if jid]
    if not job_list:
        return []
    query = """
    MATCH (c:Candidate {cand_id: $cand_id})
    OPTIONAL MATCH (c)-[:HAS_SKILL]->(cs:Skill)
    WITH c, collect(DISTINCT cs.name) AS candSkillsRaw
    WITH c, [s IN candSkillsRaw WHERE s IS NOT NULL] AS candSkills
    MATCH (j:JobPost)-[:REQUIRES_SKILL]->(s:Skill)
    WHERE j.job_id IN $job_ids AND NOT s.name IN candSkills
    RETURN s.name AS skill,
           count(DISTINCT j) AS jobCount
    ORDER BY jobCount DESC, skill ASC
    LIMIT $limit
    """
    drv = driver or get_driver()
    with drv.session() as session:
        res = session.run(query, cand_id=cand_id, job_ids=job_list, limit=limit)
        return [r.data() for r in res if r.get("skill")]
