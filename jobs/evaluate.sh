#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
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
CAL=200
SEED=0
PREFILL="--prefill"   # use prefill-only attention; remove to use generate mode

# Heads per dataset (update these from head selection output)
BGL_HEADS='[[0, 4], [0, 11], [1, 10], [0, 21], [0, 12], [0, 20], [0, 9], [0, 1], [0, 31], [0, 0]]'
SPIRIT_HEADS='[[0, 12], [0, 11], [0, 20], [0, 15], [0, 10], [0, 1], [25, 23], [0, 0], [0, 31], [0, 9]]'
THUNDERBIRD_HEADS='[[1,15],[5,4],[18,21],[20,1],[30,11]]'
HDFS_HEADS='[[0, 18], [4, 18], [13, 1], [1, 10], [0, 20], [0, 21], [9, 29], [0, 0], [5, 14], [4, 17]]'

# ==========================================================
# BGL
# ==========================================================
echo ""
echo "===== ${MODEL} - BGL - Evaluate ====="
python -u run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name bgl \
    --windowed \
    --train_csv ./data/bgl/train.csv \
    --test_csv ./data/bgl/test.csv \
    --n_cal_samples ${CAL} \
    --instruction "${INSTRUCTION}" \
    --heads "${BGL_HEADS}" \
    --seed ${SEED} \
    ${PREFILL}

# ==========================================================
# Spirit
# ==========================================================
echo ""
echo "===== ${MODEL} - Spirit - Evaluate ====="
python -u run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name spirit \
    --windowed \
    --train_csv ./data/spirit/train.csv \
    --test_csv ./data/spirit/test.csv \
    --n_cal_samples ${CAL} \
    --instruction "${INSTRUCTION}" \
    --heads "${SPIRIT_HEADS}" \
    --seed ${SEED} \
    ${PREFILL}

# ==========================================================
# Thunderbird
# ==========================================================
# echo ""
# echo "===== ${MODEL} - Thunderbird - Evaluate ====="
# python -u run_dataset.py \
#     --model_name ${MODEL} \
#     --dataset_name thunderbird \
#     --windowed \
#     --train_csv ./data/thunderbird/train.csv \
#     --test_csv ./data/thunderbird/test.csv \
#     --n_cal_samples ${CAL} \
#     --instruction "${INSTRUCTION}" \
#     --heads "${THUNDERBIRD_HEADS}" \
#     --seed ${SEED} \
#     ${PREFILL}

# ==========================================================
# HDFS
# ==========================================================
echo ""
echo "===== ${MODEL} - HDFS - Evaluate ====="
python -u run_dataset.py \
    --model_name ${MODEL} \
    --dataset_name hdfs \
    --windowed \
    --train_csv ./data/hdfs/train.csv \
    --test_csv ./data/hdfs/test.csv \
    --n_cal_samples ${CAL} \
    --instruction "${INSTRUCTION}" \
    --heads "${HDFS_HEADS}" \
    --seed ${SEED} \
    ${PREFILL}

echo ""
echo "===== Done! Results saved to ./result/<dataset>/ ====="
