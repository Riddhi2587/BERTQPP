import pyterrier as pt
import pandas as pd
import json
from pyterrier.measures import *

# if not pt.started():
#     pt.init()

pt.java.init()

def compute_perquery_scores(res_file, qrels_file, out_json, metrics=[RR@10, AP, nDCG@20]):
    # Load qrels
    qrels = pt.io.read_qrels(qrels_file)

    # Load run
    run_df = pd.read_csv(res_file, sep="\s+", names=["qid", "iter", "docno", "rank", "score", "runid"])

    # Evaluate
    eval_dict = pt.Evaluate(run_df, qrels, metrics=metrics, perquery=True)

    # Convert to JSON structure
    out = {}
    for qid, scores in eval_dict.items():
        out[qid] = {}
        for metric, val in scores.items():
            out[qid][metric] = float(val)

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved per-query scores to {out_json}")
