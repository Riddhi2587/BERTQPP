import argparse
import json
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

pt.java.init()  # ensure Java is started

def compute_scores(res_file, qrels_file, out_file, metrics=[AP, RR@10, nDCG@20]):
    # Load run file
    run_df = pd.read_csv(
        res_file, 
        sep=r"\s+", 
        names=["qid", "iter", "docno", "rank", "score", "runid"]
    )

    # Load qrels
    qrels = pt.io.read_qrels(qrels_file)

    # Evaluate with per-query scores
    eval = pt.Evaluate(run_df, qrels, metrics=metrics, perquery=True)

    # Convert to JSON-serializable
    out = {}
    for qid, scores in eval.items():
        out[qid] = {str(metric): float(val) for metric, val in scores.items()}

    # Save JSON
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved per-query scores to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=str, required=True, help="Path to .res file")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    compute_scores(args.res, args.qrels, args.out)
