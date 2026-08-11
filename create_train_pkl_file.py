import argparse
import pickle
from collections import defaultdict


def build_train_pkl(collection_path, query_map_path, run_path, output_path):
    # Load collection (docid -> doctext)
    col_dic = defaultdict(str)
    with open(collection_path, "r") as f:
        for line in f:
            docid, doctext = line.rstrip("\n").split("\t")
            col_dic[docid] = doctext

    # Load per-query text + performance label (qid <tab> qtext <tab> performance)
    q_map_dic = {}
    with open(query_map_path, "r") as f:
        for line in f:
            qid, qtext, performance = line.rstrip("\n").split("\t")
            q_map_dic[qid] = {"qtext": qtext, "performance": float(performance)}

    # Attach the top-ranked doc's text per query from the run file (qid <tab> docid <tab> rank)
    with open(run_path, "r") as f:
        for line in f:
            qid, docid, rank = line.rstrip("\n").split("\t")
            if qid in q_map_dic:
                q_map_dic[qid]["doc_text"] = col_dic.get(docid, "")

    # Drop queries that never got a doc_text (missing from the run file)
    q_map_dic = {qid: v for qid, v in q_map_dic.items() if "doc_text" in v}

    with open(output_path, "wb") as f:
        pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Saved {len(q_map_dic)} training examples to {output_path}")
    return q_map_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build query-to-doc-text-and-label map for training and save as pickle.")
    parser.add_argument("--collection", type=str, required=True, help="Path to collection.tsv")
    parser.add_argument("--query-map", type=str, required=True, help="Output of create_train_query.py (qid <tab> qtext <tab> performance)")
    parser.add_argument("--run", type=str, required=True, help="Run file with the top-ranked doc per query (qid <tab> docid <tab> rank)")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")

    args = parser.parse_args()

    build_train_pkl(args.collection, args.query_map, args.run, args.output)
