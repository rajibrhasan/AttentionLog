#!/bin/bash
# Run hyperparameter / ablation grid search.
# Sweeps (instruction, num_data, head_method) and reports best AUC per model+dataset.
#
# Usage:  bash scripts/search_ablation.sh <model_name> <dataset>
#         dataset ∈ {bgl, spirit, thunderbird}
#
# Edit DEFAULT_INSTRUCTIONS / DEFAULT_NUM_DATA_VALUES / HEAD_METHODS in
# search_hyperparams.py to change the sweep grid.
set -e

MODEL=${1:?Usage: bash scripts/search_ablation.sh <model_name> <dataset>}
DATASET=${2:?Usage: bash scripts/search_ablation.sh <model_name> <dataset>}

case "${DATASET}" in
    bgl|spirit|thunderbird) ;;
    *) echo "Unknown dataset: ${DATASET}"; exit 1 ;;
esac

OUT="./result/search/${DATASET}/${MODEL}_search.json"

python -u search_hyperparams.py \
    --model_name ${MODEL} \
    --dataset_name ${DATASET} \
    --seed 42 \
    --n_cal 200 \
    --n_test_normal 500 \
    --n_test_anomaly 500 \
    --output_path ${OUT}

echo ""
echo "Done. Results saved to ${OUT}"
