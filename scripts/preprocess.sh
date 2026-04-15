#!/bin/bash
# Preprocess raw logs into windowed train/test CSVs.
# Usage:  bash scripts/preprocess.sh <dataset>
#         dataset ∈ {bgl, spirit, thunderbird}
set -e

DATASET=${1:?Usage: bash scripts/preprocess.sh <bgl|spirit|thunderbird>}

case "${DATASET}" in
    bgl)         LOG_NAME="BGL.log" ;;
    spirit)      LOG_NAME="spirit2_5m.log" ;;
    thunderbird) LOG_NAME="Thunderbird.log" ;;
    *) echo "Unknown dataset: ${DATASET}"; exit 1 ;;
esac

echo "===== Preprocessing ${DATASET} (${LOG_NAME}) ====="
cd prepare_data && python sliding_window.py \
    --data_dir ../data/${DATASET} \
    --log_name ${LOG_NAME} \
    --dataset ${DATASET}
cd ..

echo "Done. Windowed CSVs at data/${DATASET}/{train,test}.csv"
