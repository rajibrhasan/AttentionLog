#!/bin/bash
#SBATCH --job-name=find_heads
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
# Config
# ==========================================================
MODEL="llama3_8b-attn"
INSTRUCTION="Summarize this standard server log entry"
NUM_DATA=50
PREFILL="--prefill"   # use prefill-only attention; remove to use generate mode
# ==========================================================
# BGL
# ==========================================================
echo ""
echo "===== ${MODEL} - BGL - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data ${NUM_DATA} \
    --dataset bgl \
    --windowed \
    --train_csv ./data/bgl/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL}

# ==========================================================
# Spirit
# ==========================================================
echo ""
echo "===== ${MODEL} - Spirit - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data ${NUM_DATA} \
    --dataset spirit \
    --windowed \
    --train_csv ./data/spirit/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL}

# ==========================================================
# Thunderbird
# ==========================================================
# echo ""
# echo "===== ${MODEL} - Thunderbird - Head Selection ====="
# python -u select_head.py \
#     --model_name ${MODEL} \
#     --num_data ${NUM_DATA} \
#     --dataset thunderbird \
#     --windowed \
#     --train_csv ./data/thunderbird/train.csv \
#     --instruction "${INSTRUCTION}" \
#      ${PREFILL}

# ==========================================================
# HDFS
# ==========================================================
echo ""
echo "===== ${MODEL} - HDFS - Head Selection ====="
python -u select_head.py \
    --model_name ${MODEL} \
    --num_data ${NUM_DATA} \
    --dataset hdfs \
    --windowed \
    --train_csv ./data/hdfs/train.csv \
    --instruction "${INSTRUCTION}" \
    ${PREFILL}
