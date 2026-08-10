import pandas as pd

def join_queries_with_map(map_file, query_file, output_file):
    # Load per-query MAP values
    map_df = pd.read_csv(map_file)

    # Load queries file (qid \t query)
    queries_df = pd.read_csv(query_file, sep="\t", names=["qid", "query"])

    # Merge on qid
    merged = pd.merge(queries_df, map_df, on="qid", how="inner")

    # Keep only qid, query, MAP@10
    merged = merged[["qid", "query", "RR@10"]]

    # Format MAP@10 to 4 decimal places
    merged["RR@10"] = merged["RR@10"].map(lambda x: f"{x:.4f}")

    # Save as TSV
    merged.to_csv(output_file, sep="\t", index=False, header=False)

    print(f"[INFO] Saved merged file to {output_file}")
    return merged


if __name__ == "__main__":
    map_file = "/media/pbclab/Elements/qpp/BERTQPP/perquery_mrr@10.csv"     # perquery_map@10.csv"
    query_file = "queries.train.tsv"
    output_file = "msmarco_train_query_mrr_10.tsv"

    df = join_queries_with_map(map_file, query_file, output_file)
    print(df.head())
