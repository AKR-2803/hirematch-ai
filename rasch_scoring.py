#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Guttman/Rasch-style difficulty and ability scores from the Neo4j Knowledge Graph.

This script performs the following:

1) Skill Difficulty (Rasch-style item difficulty)
   For each Skill, compute how many candidates possess it.
   Use a smoothed frequency estimate to compute item difficulty:
       p = (n + 0.5) / (N + 1)
       beta_s = log((1 - p) / p)
   where:
       - N = total number of candidates
       - n = number of candidates with the skill
   Rare skills yield larger positive difficulty values (harder skills).
   Common skills yield negative difficulty values (easier skills).
   Stored in Neo4j as: s.rasch_difficulty

2) Candidate Ability
   For each Candidate, aggregate the negative of the difficulty of the skills they possess:
       ability_raw(c) = mean( -beta_s )
   (Candidates possessing difficult skills receive higher ability values.)
   Then standardize:
       theta_c = (ability_raw(c) - mean) / std
   Stored in Neo4j as: c.rasch_ability

3) Job Difficulty
   For each JobPost, compute the mean difficulty of its required skills:
       beta_j = mean(beta_s)
   Stored in Neo4j as: j.rasch_difficulty

After running this script, the graph will contain:
- Skill nodes annotated with difficulty estimates
- Candidate nodes annotated with ability estimates
- JobPost nodes annotated with difficulty estimates

These values can be used to compute Rasch-style match probabilities:
    match = 1 / (1 + exp(-(theta_c - beta_j)))
"""

import argparse
import math
from typing import Dict, List

from neo4j import GraphDatabase


def fetch_candidate_count(session) -> int:
    result = session.run("MATCH (c:Candidate) RETURN count(c) AS n")
    return result.single()["n"]


def fetch_skill_candidate_counts(session) -> Dict[str, int]:
    """Return a mapping of skill_name → number_of_candidates_with_this_skill."""
    query = """
    MATCH (s:Skill)<-[:HAS_SKILL]-(c:Candidate)
    RETURN s.name AS name, count(c) AS n
    """
    counts = {}
    for record in session.run(query):
        counts[record["name"]] = record["n"]
    return counts


def compute_skill_difficulties(total_candidates: int, skill_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute Rasch-style skill difficulty for each skill using:
        p = (n + 0.5) / (N + 1)
        beta = log((1 - p) / p)
    """
    difficulties = {}
    for name, n in skill_counts.items():
        if total_candidates <= 0:
            beta = 0.0
        else:
            p = (n + 0.5) / (total_candidates + 1.0)
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            beta = math.log((1.0 - p) / p)
        difficulties[name] = beta
    return difficulties


def write_skill_difficulties(session, difficulties: Dict[str, float]) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (s:Skill {name: row.name})
    SET s.rasch_difficulty = row.beta
    """
    rows = [{"name": name, "beta": float(beta)} for name, beta in difficulties.items()]
    session.run(query, rows=rows)


def fetch_candidate_skills(session) -> Dict[str, List[str]]:
    """Return { candidate_id: [skill_names] }."""
    query = """
    MATCH (c:Candidate)
    OPTIONAL MATCH (c)-[:HAS_SKILL]->(s:Skill)
    RETURN c.cand_id AS cand_id, collect(s.name) AS skills
    """
    data = {}
    for record in session.run(query):
        cand_id = record["cand_id"]
        skills = [name for name in record["skills"] if name is not None]
        data[cand_id] = skills
    return data


def compute_candidate_abilities(candidate_skills: Dict[str, List[str]],
                                skill_difficulties: Dict[str, float]) -> Dict[str, float]:
    """
    Compute raw ability:
        ability_raw = mean( -beta_s )
    Then standardize across candidates.
    """
    raw = {}
    for cand_id, skills in candidate_skills.items():
        vals = []
        for name in skills:
            beta = skill_difficulties.get(name)
            if beta is not None:
                vals.append(-beta)
        raw[cand_id] = sum(vals) / len(vals) if vals else 0.0

    if not raw:
        return raw

    mean = sum(raw.values()) / len(raw)
    var = sum((v - mean) ** 2 for v in raw.values()) / max(len(raw) - 1, 1)
    std = math.sqrt(var) if var > 0 else 1.0

    standardized = {cid: (v - mean) / std for cid, v in raw.items()}
    return standardized


def write_candidate_abilities(session, abilities: Dict[str, float]) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (c:Candidate {cand_id: row.cand_id})
    SET c.rasch_ability = row.ability
    """
    rows = [{"cand_id": cid, "ability": float(theta)} for cid, theta in abilities.items()]
    session.run(query, rows=rows)


def fetch_job_required_skills(session) -> Dict[str, List[str]]:
    """Return { job_id: [skill_names] }."""
    query = """
    MATCH (j:JobPost)
    OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(s:Skill)
    RETURN j.job_id AS job_id, collect(s.name) AS skills
    """
    data = {}
    for record in session.run(query):
        job_id = record["job_id"]
        skills = [name for name in record["skills"] if name is not None]
        data[job_id] = skills
    return data


def compute_job_difficulties(job_skills: Dict[str, List[str]],
                             skill_difficulties: Dict[str, float]) -> Dict[str, float]:
    """
    Compute mean(beta_s) for skills required by each job.
    """
    job_diffs = {}
    for job_id, skills in job_skills.items():
        vals = []
        for name in skills:
            beta = skill_difficulties.get(name)
            if beta is not None:
                vals.append(beta)
        job_diffs[job_id] = sum(vals) / len(vals) if vals else 0.0
    return job_diffs


def write_job_difficulties(session, job_difficulties: Dict[str, float]) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (j:JobPost {job_id: row.job_id})
    SET j.rasch_difficulty = row.difficulty
    """
    rows = [{"job_id": jid, "difficulty": float(beta)} for jid, beta in job_difficulties.items()]
    session.run(query, rows=rows)


def main():
    ap = argparse.ArgumentParser(description="Compute Rasch-style difficulty and ability scores for Neo4j KG.")
    ap.add_argument("--uri", required=True, help="bolt://localhost:7687")
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as session:
        total_cands = fetch_candidate_count(session)

        skill_counts = fetch_skill_candidate_counts(session)
        skill_diffs = compute_skill_difficulties(total_cands, skill_counts)
        write_skill_difficulties(session, skill_diffs)

        cand_skills = fetch_candidate_skills(session)
        cand_abilities = compute_candidate_abilities(cand_skills, skill_diffs)
        write_candidate_abilities(session, cand_abilities)

        job_skills = fetch_job_required_skills(session)
        job_diffs = compute_job_difficulties(job_skills, skill_diffs)
        write_job_difficulties(session, job_diffs)

    driver.close()
    print("Rasch-style difficulty and ability scores written to Neo4j.")


if __name__ == "__main__":
    main()
