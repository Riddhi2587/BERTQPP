# import collections,pickle
# from collections import defaultdict
# from typing import runtime_checkable
# col_dic=defaultdict(list)

# collection_file=open('/media/pbclab/Elements/qpp/msmarco_dataset/Passage_Retrieval/collection/collection.tsv','r').readlines()

# for line in collection_file:
#     docid,doctext= line.rstrip().split('\t')
#     col_dic[docid]=doctext
# q_file= open ('trecdl1920_queries.tsv','r').readlines()

# q_map_dic={}


# for line in q_file:
#     qid,qtext=line.rstrip().split('\t')
#     q_map_dic[qid]={}
#     q_map_dic[qid] ["qtext"]=qtext
    
# run_file=open('run/bm25_first_docs_dev.tsv','r').readlines()
# for line in run_file:
#     qid,docid,rank=line.split('\t')
#     if qid in q_map_dic.keys():
#         q_map_dic[qid]["doc_text"]=col_dic[docid]


# with open('pklfiles/test_dev_map.pkl', 'wb') as f:
#     pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)

import collections,pickle
import argparse
import pickle
from collections import defaultdict

def build_qid_doc_map(collection_path, query_path, run_path, output_path):
    # Load collection (docid -> doctext)
    col_dic = defaultdict(str)
    with open(collection_path, "r") as f:
        for line in f:
            docid, doctext = line.rstrip().split("\t")
            col_dic[docid] = doctext

    # Load queries
    q_map_dic = {}
    with open(query_path, "r") as f:
        for line in f:
            qid, qtext = line.rstrip().split("\t")
            q_map_dic[qid] = {"qtext": qtext}

    # Load run file and attach doc text
    with open(run_path, "r") as f:
        for line in f:
            qid, docid, rank = line.strip().split("\t")
            if qid in q_map_dic:
                q_map_dic[qid]["doc_text"] = col_dic.get(docid, "")

    # Save pickle
    with open(output_path, "wb") as f:
        pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved pickle file to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build query-to-doc-text map and save as pickle.")
    parser.add_argument("--collection", type=str, required=True,
                        help="Path to collection.tsv")
    parser.add_argument("--queries", type=str, required=True,
                        help="Path to queries file (qid <tab> query_text)")
    parser.add_argument("--run", type=str, required=True,
                        help="Path to run file (qid <tab> docid <tab> rank)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pickle file path")

    args = parser.parse_args()

    build_qid_doc_map(args.collection, args.queries, args.run, args.output)



# python3 create_test_pkl_files.py --collection /media/pbclab/Elements/qpp/msmarco_dataset/Passage_Retrieval/collection/collection.tsv --queries trecdl1920_queries.tsv --run run/result_files/*.res --output pklfiles/trecdl_1920/