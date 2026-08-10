# import pickle 
# from scipy.stats import kendalltau,pearsonr

# from sentence_transformers.cross_encoder import CrossEncoder
# trained_model="msmarco_tuned_model-ce_bert-base-uncased_e10_b16"

# with open('pklfiles/test_dev_map.pkl', 'rb') as f:
#     q_map_first_doc_test=pickle.load(f)

# sentences = []
# map_value_test=[]
# queries=[]
# for key in q_map_first_doc_test:
#     sentences.append([q_map_first_doc_test[key]["qtext"],q_map_first_doc_test[key]["doc_text"]])
#     queries.append(key)

# model = CrossEncoder("models/"+trained_model, num_labels=1)
# scores=model.predict(sentences)
# actual=[]
# predicted=[]
# out=open('results/QPP-corss_'+trained_model,'w')
# for i in range(len(sentences)):
#     predicted.append(float(scores[i]))
#     out.write(queries[i]+'\t'+str(predicted[i])+'\n')
# out.close()


import argparse
import pickle
from sentence_transformers.cross_encoder import CrossEncoder

def run_crossencoder(pkl_file, model_path, output_file):
    # Load pickle
    with open(pkl_file, "rb") as f:
        q_map_first_doc_test = pickle.load(f)

    # Prepare sentences
    sentences = []
    queries = []
    for key in q_map_first_doc_test:
        try:
            sentences.append([q_map_first_doc_test[key]["qtext"], q_map_first_doc_test[key]["doc_text"]])
            queries.append(key)
        except:
            sentences.append([q_map_first_doc_test[key]["qtext"]])
            queries.append(key)

    # Load trained model
    model = CrossEncoder(model_path, num_labels=1)

    # Predict scores
    scores = model.predict(sentences)

    # Save results
    with open(output_file, "w") as out:
        for i in range(len(sentences)):
            out.write(f"{queries[i]}\t{float(scores[i])}\n")

    print(f"[INFO] Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossEncoder QPP prediction script")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Input pickle file (query-doc map)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained CrossEncoder model")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file to save predictions")

    args = parser.parse_args()

    run_crossencoder(args.pkl, args.model, args.output)
