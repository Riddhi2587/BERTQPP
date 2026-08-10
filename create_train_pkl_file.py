import pickle
from collections import defaultdict
col_dic=defaultdict(list)

# replace msmarco collection path
collection_file=open('/media/pbclab/Elements/qpp/msmarco_dataset/Passage_Retrieval/collection/collection.tsv','r').readlines()

for line in collection_file:
    docid,doctext= line.rstrip().split('\t')
    col_dic[docid]=doctext
q_map_dic_train={}
q_map_dic_test={}

# path to evaluation metric per query
# qid<\t>metric_value
query_map_file=open('msmarco_train_query_mrr_10.tsv','r').readlines()
for line in query_map_file:
    qid,qtext,qmap=line.split('\t')
    q_map_dic_train[qid]={}
    q_map_dic_train[qid] ["qtext"]=qtext
    q_map_dic_train[qid] ["performance"]=float(qmap)

# run file including first retrieved documents per query
run_file=open('/media/pbclab/1o9SSD/payel/qpp/msmarco/bm25_ret/bm25_first_docs_train.tsv','r').readlines()
for line in run_file:
    qid,docid,rank=line.split('\t')
    if qid in q_map_dic_train.keys():
        q_map_dic_train[qid]["doc_text"]=col_dic[docid]

with open('/media/pbclab/Elements/qpp/BERTQPP/pklfiles/msmarco_train_mrr_10.pkl', 'wb') as f:
    pickle.dump(q_map_dic_train, f, pickle.HIGHEST_PROTOCOL)

