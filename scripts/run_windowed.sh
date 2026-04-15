#!/bin/bash
# Find heads + evaluate Attention-Log on windowed log sequences.
# Usage:  bash scripts/run_windowed.sh <model_name> <dataset>
#         dataset ∈ {bgl, spirit, thunderbird}
set -e

MODEL=${1:?Usage: bash scripts/run_windowed.sh <model_name> <dataset>}
DATASET=${2:?Usage: bash scripts/run_windowed.sh <model_name> <dataset>}

DATA_DIR="./data/${DATASET}"
TRAIN_CSV="${DATA_DIR}/train.csv"
TEST_CSV="${DATA_DIR}/test.csv"
INSTRUCTION="Repeat this normal log entry exactly"

echo "===== ${MODEL} - ${DATASET} - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data 200 \
    --dataset ${DATASET} \
    --windowed \
    --train_csv ${TRAIN_CSV} \
    --instruction "${INSTRUCTION}"

echo ""
echo "===== ${MODEL} - ${DATASET} - Evaluation ====="
python run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name ${DATASET} \
    --windowed \
    --train_csv ${TRAIN_CSV} \
    --test_csv ${TEST_CSV} \
    --n_cal_samples 200 \
    --instruction "${INSTRUCTION}" \
    --seed 0
