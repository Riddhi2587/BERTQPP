import argparse
import csv
from collections import defaultdict
import pytrec_eval


def parse_run(path, cutoff):
    """TREC-style run file -> {qid: {docid: score}}, keeping only the top `cutoff` docs per query by rank."""
    ranked = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            ranked[qid].append((int(rank), docid, float(score)))

    run = {}
    for qid, docs in ranked.items():
        docs.sort(key=lambda x: x[0])
        run[qid] = {docid: score for _, docid, score in docs[:cutoff]}
    return run


def compute_perquery_metric(runfile, qrelsfile, metric="MRR", cutoff=10):
    """
    Compute per-query MRR@cutoff or MAP@cutoff using pytrec_eval.

    Args:
        runfile (str): Path to the TREC-style run file.
        qrelsfile (str): Path to the qrels file.
        metric (str): "MRR" or "MAP".
        cutoff (int): Cutoff for the metric (default: 10).

    Returns:
        dict: {qid: metric_value}
    """
    with open(qrelsfile) as f:
        qrels = pytrec_eval.parse_qrel(f)

    run = parse_run(runfile, cutoff)

    measure_name = "recip_rank" if metric == "MRR" else f"map_cut_{cutoff}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {measure_name})
    results = evaluator.evaluate(run)

    return {qid: scores[measure_name] for qid, scores in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-query MRR/MAP@cutoff from a run file and qrels (via pytrec_eval).")
    parser.add_argument("--run", type=str, required=True, help="Path to the TREC-style run file")
    parser.add_argument("--qrels", type=str, required=True, help="Path to the qrels file")
    parser.add_argument("--metric", type=str, default="MRR", choices=["MRR", "MAP"], help="Metric to compute (default: MRR)")
    parser.add_argument("--cutoff", type=int, default=10, help="Cutoff for the metric (default: 10)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path (columns: qid, {metric}@{cutoff})")

    args = parser.parse_args()

    per_query = compute_perquery_metric(args.run, args.qrels, metric=args.metric, cutoff=args.cutoff)
    metric_col = f"{args.metric}@{args.cutoff}"

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", metric_col])
        for qid, value in per_query.items():
            writer.writerow([qid, value])

    print(f"[INFO] Saved per-query {metric_col} for {len(per_query)} queries to {args.output}")
