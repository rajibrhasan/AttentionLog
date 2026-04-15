#!/bin/bash
# Run head selection on BGL, Liberty (spirit), and Thunderbird for one model.
# Usage:  bash scripts/find_heads_all.sh <model_name>
set -e

MODEL=${1:?Usage: bash scripts/find_heads_all.sh <model_name>}
INSTRUCTION="Summarize this standard server log entry"
NUM_DATA=50

echo "===== ${MODEL} - BGL - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data ${NUM_DATA} \
    --dataset bgl \
    --bgl_path ./data/bgl/BGL.log \
    --instruction "${INSTRUCTION}"

echo ""
echo "===== ${MODEL} - Liberty - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data ${NUM_DATA} \
    --dataset spirit \
    --spirit_path ./data/spirit/spirit2_5m.log \
    --instruction "${INSTRUCTION}"

echo ""
echo "===== ${MODEL} - Thunderbird - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data ${NUM_DATA} \
    --dataset thunderbird \
    --thunderbird_path ./data/thunderbird/Thunderbird.log \
    --instruction "${INSTRUCTION}"

echo ""
echo "Done. Copy the printed top-k heads into scripts/run_dataset_all.sh."
