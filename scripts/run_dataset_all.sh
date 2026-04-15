#!/bin/bash
# Evaluate one model on BGL, Liberty (spirit), and Thunderbird using the
# heads previously found by find_heads_all.sh.
# Usage:  bash scripts/run_dataset_all.sh <model_name>
set -e

MODEL=${1:?Usage: bash scripts/run_dataset_all.sh <model_name>}
INSTRUCTION="Summarize this standard server log entry"
CAL=200
SEED=0

# Heads per model per dataset (paste output of find_heads_all.sh here).
if [ "${MODEL}" = "llama3_8b-attn" ]; then
    BGL_HEADS='[[22, 10], [20, 26], [28, 4], [14, 12], [1, 11], [22, 16], [2, 24], [10, 22], [21, 15], [9, 29]]'
    SPIRIT_HEADS='[[20, 10], [20, 9], [22, 1], [20, 1], [25, 15], [1, 15], [24, 25], [30, 11], [18, 21], [5, 4]]'
    THUNDERBIRD_HEADS='[[0, 1], [27, 27], [9, 1], [23, 28], [0, 30], [27, 11], [10, 28], [2, 12], [0, 16], [10, 30]]'
elif [ "${MODEL}" = "qwen2-attn" ]; then
    BGL_HEADS='[[23, 4], [4, 5], [20, 3], [22, 8], [25, 8], [11, 7], [10, 2], [21, 3], [27, 10], [15, 10], [23, 3], [12, 8], [17, 2], [20, 5], [20, 7], [14, 10]]'
    SPIRIT_HEADS='[[2, 1], [10, 6], [14, 4], [13, 10], [9, 0], [13, 4], [14, 10], [1, 9], [24, 6], [11, 7], [17, 2], [10, 2], [18, 2], [10, 5], [6, 7], [15, 3]]'
    THUNDERBIRD_HEADS='[[18, 8], [3, 5], [17, 5], [8, 4], [6, 10], [11, 10], [18, 11], [12, 7], [12, 11], [19, 5], [20, 1], [27, 7], [6, 8], [3, 3], [15, 1], [18, 6]]'
elif [ "${MODEL}" = "granite3_8b-attn" ]; then
    BGL_HEADS='[[30, 26], [10, 30], [6, 14], [8, 14], [22, 18], [37, 9], [0, 24], [16, 14], [17, 21], [0, 9], [0, 4], [37, 28]]'
    SPIRIT_HEADS='[[8, 14], [16, 19], [10, 5], [11, 10], [10, 30], [38, 20], [0, 9], [17, 21], [16, 14], [0, 24], [0, 4], [25, 9]]'
    THUNDERBIRD_HEADS='[[14, 30], [10, 10], [30, 12], [9, 15], [30, 15], [10, 29], [10, 26], [18, 7], [10, 25], [31, 11], [17, 16], [9, 30]]'
elif [ "${MODEL}" = "mistral_7b-attn" ]; then
    BGL_HEADS='[[30, 4], [12, 4], [21, 22], [20, 18], [12, 7], [23, 0], [4, 7], [9, 25], [30, 20], [4, 20]]'
    SPIRIT_HEADS='[[24, 26], [12, 4], [30, 4], [23, 0], [20, 12], [13, 23], [30, 10], [12, 7], [9, 0], [3, 14]]'
    THUNDERBIRD_HEADS='[[23, 6], [27, 2], [4, 14], [23, 7], [9, 9], [3, 19], [9, 28], [4, 21], [11, 22], [6, 20]]'
elif [ "${MODEL}" = "gemma2_9b-attn" ]; then
    BGL_HEADS='[[13, 6], [40, 6], [23, 7], [19, 6], [18, 5], [14, 15], [20, 5], [18, 4], [25, 8], [1, 12], [12, 9], [32, 15], [28, 13], [26, 14], [23, 11], [16, 5], [24, 0], [27, 9], [22, 11], [12, 8], [9, 1], [9, 7], [16, 9], [14, 4], [10, 11], [8, 12], [1, 3], [30, 1], [9, 2], [9, 3], [15, 11], [28, 10], [19, 5]]'
    SPIRIT_HEADS='[[8, 10], [5, 1], [9, 1], [8, 2], [20, 5], [20, 2], [7, 3], [23, 14], [12, 9], [16, 12], [2, 15], [3, 5], [14, 4], [10, 11], [25, 8], [16, 5], [18, 4], [17, 14], [8, 12], [1, 3], [14, 15], [22, 8], [22, 11], [22, 10], [16, 9], [13, 7], [12, 1], [9, 7], [9, 3], [9, 2], [18, 5], [15, 11], [19, 5]]'
    THUNDERBIRD_HEADS='[[9, 6], [2, 15], [11, 14], [34, 5], [31, 1], [2, 4], [0, 13], [4, 6], [29, 15], [5, 8], [14, 5], [7, 6], [31, 3], [4, 7], [6, 1], [6, 13], [32, 6], [16, 1], [6, 10], [12, 13], [10, 8], [5, 15], [5, 9], [5, 5], [40, 4], [11, 5], [4, 15], [12, 4], [8, 0], [15, 12], [0, 5], [2, 8], [16, 11]]'
else
    echo "ERROR: No heads configured for model ${MODEL}. Run find_heads_all.sh first."
    exit 1
fi

echo "===== ${MODEL} - BGL - Evaluate ====="
python -u run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name bgl \
    --bgl_path ./data/bgl/BGL.log \
    --n_cal_samples ${CAL} \
    --instruction "${INSTRUCTION}" \
    --heads "${BGL_HEADS}" \
    --seed ${SEED}

echo ""
echo "===== ${MODEL} - Liberty - Evaluate ====="
python -u run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name spirit \
    --spirit_path ./data/spirit/spirit2_5m.log \
    --n_cal_samples ${CAL} \
    --instruction "${INSTRUCTION}" \
    --heads "${SPIRIT_HEADS}" \
    --seed ${SEED}

echo ""
echo "===== ${MODEL} - Thunderbird - Evaluate ====="
python -u run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name thunderbird \
    --thunderbird_path ./data/thunderbird/Thunderbird.log \
    --n_cal_samples ${CAL} \
    --instruction "${INSTRUCTION}" \
    --heads "${THUNDERBIRD_HEADS}" \
    --seed ${SEED}

echo ""
echo "===== Done! Results saved to ./result/<dataset>/ ====="
