#!/bin/bash
set -e

MODEL=${1:?Usage: bash scripts/visualize_all.sh <model_name>}
INSTRUCTION="Summarize this standard server log entry"
NUM_DATA=50

echo "===== Visualize ${MODEL} - BGL ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset bgl \
    --num_data ${NUM_DATA} \
    --bgl_path ./data/bgl/BGL.log \
    --instruction "${INSTRUCTION}" \
    --save_dir ./plots/${MODEL}/bgl

echo ""
echo "===== Visualize ${MODEL} - Spirit ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset spirit \
    --num_data ${NUM_DATA} \
    --spirit_path ./data/spirit/spirit2_5m.log \
    --instruction "${INSTRUCTION}" \
    --save_dir ./plots/${MODEL}/spirit

echo ""
echo "===== Visualize ${MODEL} - Thunderbird ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset thunderbird \
    --num_data ${NUM_DATA} \
    --thunderbird_path ./data/thunderbird/Thunderbird.log \
    --instruction "${INSTRUCTION}" \
    --save_dir ./plots/${MODEL}/thunderbird

echo ""
echo "===== Done! Plots saved to ./plots/${MODEL}/ ====="
