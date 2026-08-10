#!/bin/bash

COLLECTION="/media/pbclab/Elements/qpp/msmarco_dataset/Passage_Retrieval/collection/collection.tsv"
QUERIES="trecdl2019_queries.tsv"
RUN_DIR="/media/pbclab/Elements/qpp/BERTQPP/run/result_files/2019"
OUT_DIR="/media/pbclab/Elements/qpp/BERTQPP/pklfiles/trecdl_19"

mkdir -p "$OUT_DIR"

for runfile in "$RUN_DIR"/*.tsv; do
    base=$(basename "$runfile" .tsv)   # get filename without .res
    outfile="$OUT_DIR/${base}.pkl"
    echo "[INFO] Processing $runfile -> $outfile"

    python3 create_test_pkl_files.py \
        --collection "$COLLECTION" \
        --queries "$QUERIES" \
        --run "$runfile" \
        --output "$outfile"
done
