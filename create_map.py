import pyterrier as pt
import pandas as pd

if not pt.started():
    pt.init()

def compute_perquery_map(runfile, qrelsfile, cutoff=10):
    """
    Compute per-query MAP@cutoff for a given run file and qrels file.

    Args:
        runfile (str): Path to the TREC .res run file.
        qrelsfile (str): Path to the TREC qrels file.
        cutoff (int): Cutoff for MAP (default: 10 for MAP@10).

    Returns:
        pandas.DataFrame: Per-query MAP scores.
    """
    # Load qrels
    qrels = pt.io.read_qrels(qrelsfile)

    # Load run file
    run_df = pd.read_csv(
        runfile,
        sep=r"\s+",
        names=["qid", "iter", "docno", "rank", "score", "runid"]
    )

    # Evaluate per-query MAP@cutoff
    results = pt.Evaluate(run_df, qrels, metrics=[pt.measures.MRR @ cutoff], perquery=True)
    # results = pt.Evaluate(run_df, qrels, metrics=[pt.measures.AP @ cutoff], perquery=True)

    # Convert to DataFrame
    perquery_df = pd.DataFrame.from_dict(results, orient="index")
    perquery_df.reset_index(inplace=True)
    perquery_df.rename(columns={"index": "qid", f"MRR@{cutoff}": f"MRR@{cutoff}"}, inplace=True)

    return perquery_df

# Example usage
if __name__ == "__main__":
    runfile = "/media/pbclab/1o9SSD/payel/qpp/msmarco/bm25_ret/msmarco.run-bm25-train-1000ret.txt"
    qrelsfile = "/media/pbclab/Elements/qpp/msmarco_dataset/Passage_Retrieval/msmarco.qrels-train.clean.txt"

    perquery_map = compute_perquery_map(runfile, qrelsfile, cutoff=10)
    print(perquery_map.head())
    perquery_map.to_csv("/media/pbclab/Elements/qpp/BERTQPP/perquery_mrr@10.csv", index=False)
    # print("[INFO] Saved per-query MAP@10 to perquery_map@50.csv")
    print("[INFO] Saved per-query MAP@10 to perquery_mrr@10.csv")
