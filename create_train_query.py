import argparse
import pandas as pd


def join_queries_with_map(map_file, query_file, metric_col, output_file):
    # Load per-query metric values
    map_df = pd.read_csv(map_file)
    map_df["qid"] = map_df["qid"].astype(str)

    # Load queries file (qid \t query)
    queries_df = pd.read_csv(query_file, sep="\t", names=["qid", "query"], dtype={"qid": str})

    # Merge on qid
    merged = pd.merge(queries_df, map_df, on="qid", how="inner")

    # Keep only qid, query, metric_col
    merged = merged[["qid", "query", metric_col]]

    # Format metric to 4 decimal places
    merged[metric_col] = merged[metric_col].map(lambda x: f"{x:.4f}")

    # Save as TSV
    merged.to_csv(output_file, sep="\t", index=False, header=False)

    print(f"[INFO] Saved {len(merged)} queries with {metric_col} to {output_file}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join per-query metric values with query text.")
    parser.add_argument("--map-file", type=str, required=True, help="CSV from create_map.py (columns: qid, {metric}@{cutoff})")
    parser.add_argument("--query-file", type=str, required=True, help="Queries file (qid <tab> query_text)")
    parser.add_argument("--metric-col", type=str, required=True, help="Metric column name to keep, e.g. MRR@10")
    parser.add_argument("--output", type=str, required=True, help="Output TSV path (qid <tab> query <tab> metric)")

    args = parser.parse_args()

    join_queries_with_map(args.map_file, args.query_file, args.metric_col, args.output)
