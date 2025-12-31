"""
evaluation.py

Evaluate the unified ranking engine using a labeled ground-truth file.

Ground truth format (CSV):

    cand_id,job_id,rel
    CAND_001,JOB_123,1
    CAND_001,JOB_456,0
    CAND_002,JOB_999,1
    ...

Columns:
- cand_id: candidate ID (string, matches Candidate.cand_id in Neo4j)
- job_id:  job ID (string, matches JobPost.job_id in Neo4j)
- rel:     relevance label (int or float). rel > 0 == relevant.

You can evaluate two modes:
1) jobs_for_candidate   (default)
   - For each cand_id in ground truth, rank jobs using rank_jobs_for_candidate
   - Compare ranked list against the relevant job_ids in ground truth.

2) candidates_for_job
   - For each job_id in ground truth, rank candidates using rank_candidates_for_job
   - Compare ranked list against the relevant cand_ids in ground truth.

Metrics:
- Precision@K
- Recall@K
- MAP@K (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)

Usage example:

    python3 evaluation.py \\
        --ground-truth ground_truth.csv \\
        --mode jobs_for_candidate \\
        --k 10

Requires:
- ranking_engine.py in the same directory.
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

from ranking_engine import (
    rank_jobs_for_candidate,
    rank_candidates_for_job,
)


# metrics helpers

def precision_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        # no relevant items, convention: recall is 0.0 for this query
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def average_precision_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    """AP@K: average of precision@i over i where i-th item is relevant."""
    if not relevant_ids:
        return 0.0
    ap_sum = 0.0
    hit_count = 0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_ids:
            hit_count += 1
            ap_sum += hit_count / i
    if hit_count == 0:
        return 0.0
    return ap_sum / min(len(relevant_ids), k)


def dcg_at_k(gains: List[float], k: int) -> float:
    """discounted Cumulative Gain up to K given a list of gains in rank order."""
    from math import log2
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        dcg += g / log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: List[str], rel_map: Dict[str, float], k: int) -> float:
    """
    NDCG@K: rel_map is a mapping from item_id -> relevance (e.g., 0/1 or graded).
    """
    if not rel_map:
        return 0.0

    # gains in the order of our ranking
    gains = [rel_map.get(rid, 0.0) for rid in ranked_ids[:k]]
    dcg = dcg_at_k(gains, k)

    # ideal gains: sort by relevance descending
    ideal_gains = sorted(rel_map.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ----------------- I/O helpers ----------------- #

def load_ground_truth(path: str, mode: str):
    """
    returns:
      for jobs_for_candidate:
        gt_by_query: Dict[cand_id -> Dict[job_id -> rel]]
      for candidates_for_job:
        gt_by_query: Dict[job_id -> Dict[cand_id -> rel]]
    """
    gt_by_query: Dict[str, Dict[str, float]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"cand_id", "job_id", "rel"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"ground_truth CSV must have columns: {required_cols}")

        for row in reader:
            cand_id = row["cand_id"]
            job_id = row["job_id"]
            try:
                rel = float(row["rel"])
            except ValueError:
                rel = 0.0

            if mode == "jobs_for_candidate":
                # query = candidate, items = jobs
                gt_by_query[cand_id][job_id] = rel
            else:
                # mode == "candidates_for_job"
                # query = job, items = candidates
                gt_by_query[job_id][cand_id] = rel

    return gt_by_query


# eval logic 

def evaluate_jobs_for_candidate(
    gt_by_cand: Dict[str, Dict[str, float]],
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    for each cand_id in ground truth:
      - call rank_jobs_for_candidate(cand_id)
      - compare to relevant job_ids from ground truth
    """
    import statistics

    precs = []
    recs = []
    maps = []
    ndcgs = []

    for cand_id, rel_map in gt_by_cand.items():
        relevant_jobs = {jid for jid, r in rel_map.items() if r > 0.0}

        # get model ranking
        results = rank_jobs_for_candidate(
            cand_id,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        ranked_job_ids = [r["job_id"] for r in results]

        precs.append(precision_at_k(ranked_job_ids, relevant_jobs, k))
        recs.append(recall_at_k(ranked_job_ids, relevant_jobs, k))
        maps.append(average_precision_at_k(ranked_job_ids, relevant_jobs, k))
        ndcgs.append(ndcg_at_k(ranked_job_ids, rel_map, k))

    macro_prec = statistics.mean(precs) if precs else 0.0
    macro_rec = statistics.mean(recs) if recs else 0.0
    macro_map = statistics.mean(maps) if maps else 0.0
    macro_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0

    return macro_prec, macro_rec, macro_map, macro_ndcg


def evaluate_candidates_for_job(
    gt_by_job: Dict[str, Dict[str, float]],
    k: int,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    for each job_id in ground truth:
      - call rank_candidates_for_job(job_id)
      - compare to relevant cand_ids from ground truth
    """
    import statistics

    precs = []
    recs = []
    maps = []
    ndcgs = []

    for job_id, rel_map in gt_by_job.items():
        relevant_cands = {cid for cid, r in rel_map.items() if r > 0.0}

        # get model ranking
        results = rank_candidates_for_job(
            job_id,
            k=k,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        ranked_cand_ids = [r["cand_id"] for r in results]

        precs.append(precision_at_k(ranked_cand_ids, relevant_cands, k))
        recs.append(recall_at_k(ranked_cand_ids, relevant_cands, k))
        maps.append(average_precision_at_k(ranked_cand_ids, relevant_cands, k))
        ndcgs.append(ndcg_at_k(ranked_cand_ids, rel_map, k))

    macro_prec = statistics.mean(precs) if precs else 0.0
    macro_rec = statistics.mean(recs) if recs else 0.0
    macro_map = statistics.mean(maps) if maps else 0.0
    macro_ndcg = statistics.mean(ndcgs) if ndcgs else 0.0

    return macro_prec, macro_rec, macro_map, macro_ndcg


# cli
def main():
    ap = argparse.ArgumentParser(description="Evaluate KG-based matcher against ground truth.")
    ap.add_argument(
        "--ground-truth",
        required=True,
        help="Path to CSV file with columns: cand_id, job_id, rel",
    )
    ap.add_argument(
        "--mode",
        choices=["jobs_for_candidate", "candidates_for_job"],
        default="jobs_for_candidate",
        help="Evaluation mode (default: jobs_for_candidate)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank cutoff K for metrics (default: 10)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for skill coverage",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Weight for Rasch match",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Weight for TransR similarity",
    )
    args = ap.parse_args()

    print(f"📂 Loading ground truth from: {args.ground_truth}")
    gt_by_query = load_ground_truth(args.ground_truth, mode=args.mode)
    print(f"🔢 Found {len(gt_by_query)} queries with labels (mode={args.mode})")

    if args.mode == "jobs_for_candidate":
        prec, rec, m_ap, ndcg = evaluate_jobs_for_candidate(
            gt_by_query,
            k=args.k,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
    else:
        prec, rec, m_ap, ndcg = evaluate_candidates_for_job(
            gt_by_query,
            k=args.k,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )

    print("\nEvaluation Results")
    print(f"  Mode       : {args.mode}")
    print(f"  K          : {args.k}")
    print(f"  alpha/beta/gamma = {args.alpha}/{args.beta}/{args.gamma}")
    print(f"  Precision@K: {prec:.4f}")
    print(f"  Recall@K   : {rec:.4f}")
    print(f"  MAP@K      : {m_ap:.4f}")
    print(f"  NDCG@K     : {ndcg:.4f}")


if __name__ == "__main__":
    main()
