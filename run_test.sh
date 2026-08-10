#!/bin/bash
set -e

PKL_DIR="pklfiles/trecdl_19"
OUT_DIR="results/trecdl_19/rr10/"
MODEL="models/saved_model/tuned_model-ce_bert-base-uncased_e1_b8"
# MODEL="models/msmarco_tuned_model-ce_bert-base-uncased_e1_b_mrr1016_ep1"
# MODEL="models/msmarco_tuned_model-ce_bert-base-uncased_e10_b_mrr1016"  #mrr@10
# MODEL="models/msmarco_tuned_model-ce_bert-base-uncased_e10_b_map5016"  #map50
# MODEL=  "models/msmarco_tuned_model-ce_bert-base-uncased_e10_b16"  #map10

mkdir -p "$OUT_DIR"

for pkl_file in "$PKL_DIR"/*.pkl; do
    base=$(basename "$pkl_file" .pkl)
    out_file="$OUT_DIR/${base}_savedmodel.txt"

    echo "[INFO] Processing $pkl_file -> $out_file"
    python3 test_CE.py \
        --pkl "$pkl_file" \
        --model "$MODEL" \
        --output "$out_file"
done
