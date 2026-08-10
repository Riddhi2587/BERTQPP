"""
Evaluation script for QPP predictions.
Compares actual per-query effectiveness scores vs. predicted QPP scores.
"""

import argparse
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def evaluation(ap_path, pp_path, target_metric):
    # Load actual retrieval scores
    with open(ap_path, 'r') as f:
        ap_bank = json.load(f)

    # Extract per-query target metric
    ap = {qid: float(metrics[target_metric]) for qid, metrics in ap_bank.items() if target_metric in metrics}

    # Load predicted QPP scores
    pp = {}
    with open(pp_path, 'r') as f:
        for line in f:
            qid, val = line.strip().split()
            pp[qid] = float(val)

    # Align queries
    common_qids = sorted(set(ap.keys()) & set(pp.keys()))
    ap_list = [ap[qid] for qid in common_qids]
    pp_list = [pp[qid] for qid in common_qids]

    print(f"[INFO] Target metric = {target_metric}")
    print(f"[INFO] #queries in actual = {len(ap)}, in predicted = {len(pp)}, in common = {len(common_qids)}")
    print(f"[INFO] Mean {target_metric} = {np.mean(ap_list):.3f}")

    # Compute correlations
    pearson_corr, _ = pearsonr(ap_list, pp_list)
    spearman_corr, _ = spearmanr(ap_list, pp_list)
    kendall_corr, _ = kendalltau(ap_list, pp_list)

    results = {
        "Pearson": round(pearson_corr, 3),
        "Spearman": round(spearman_corr, 3),
        "Kendall": round(kendall_corr, 3),
    }

    print("[RESULTS]", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True, help="Path to JSON with per-query actual scores")
    parser.add_argument("--predicted", type=str, required=True, help="Path to predicted scores (qid score)")
    parser.add_argument("--target_metric", type=str, default="AP", help="Metric to evaluate (e.g., AP, RR@10, nDCG@20)")
    args = parser.parse_args()

    evaluation(args.actual, args.predicted, args.target_metric)
