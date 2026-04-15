#!/bin/bash
#SBATCH --job-name=visualize_heads
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# ==========================================================
# Environment setup
# ==========================================================
source ~/Attention-Tracker/.venv/bin/activate

echo "Node: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

cd $HOME/Attention-Tracker

# ==========================================================
# Config - change these as needed
# ==========================================================
MODEL="llama3_8b-attn"
INSTRUCTION="Summarize this standard server log entry"
NUM_DATA=50
PREFILL="--prefill"   # use prefill-only attention; remove to use generate mode

# ==========================================================
# BGL
# ==========================================================
echo ""
echo "===== Visualize ${MODEL} - BGL ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset bgl \
    --num_data ${NUM_DATA} \
    --windowed \
    --train_csv ./data/bgl/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL} \
    --save_dir ./plots/${MODEL}/bgl

# ==========================================================
# Spirit
# ==========================================================
echo ""
echo "===== Visualize ${MODEL} - Spirit ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset spirit \
    --num_data ${NUM_DATA} \
    --windowed \
    --train_csv ./data/spirit/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL} \
    --save_dir ./plots/${MODEL}/spirit

# ==========================================================
# Thunderbird
# ==========================================================
echo ""
echo "===== Visualize ${MODEL} - Thunderbird ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset thunderbird \
    --num_data ${NUM_DATA} \
    --windowed \
    --train_csv ./data/thunderbird/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL} \
    --save_dir ./plots/${MODEL}/thunderbird

# ==========================================================
# HDFS
# ==========================================================
echo ""
echo "===== Visualize ${MODEL} - HDFS ====="
python -u visualize_heads.py \
    --model_name ${MODEL} \
    --dataset hdfs \
    --num_data ${NUM_DATA} \
    --windowed \
    --train_csv ./data/hdfs/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL} \
    --save_dir ./plots/${MODEL}/hdfs

echo ""
echo "===== Done! Plots saved to ./plots/${MODEL}/ ====="
