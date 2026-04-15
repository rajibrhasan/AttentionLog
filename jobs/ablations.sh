#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-13

# ==========================================================
# Ablation sweep — Attention-Log
#
#  A. Calibration size       — sweep DATASETS  (1 model × 3 datasets =  3 jobs)
#  B. Head-k                  — sweep MODELS    (5 models × 1 dataset =  5 jobs)
#  C. Instruction sensitivity — sweep DATASETS  (1 model × 3 datasets =  3 jobs)
#  D. Head selection method   — sweep DATASETS  (1 model × 3 datasets =  3 jobs)
#                                                 ─────────────────────────────
#                                                              total =  14 jobs
#
# Submit with:
#     sbatch jobs/ablations.sh
#
# Re-run a single failed task:
#     sbatch --array=7 jobs/ablations.sh
# ==========================================================

source ~/Attention-Tracker/.venv/bin/activate

echo "Node: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

cd $HOME/NLPProject/AttentionLog

# Default model used by the dataset-sweeping ablations (A, C, D).
DEFAULT_MODEL="llama3_8b-attn"

# Ablation B sweeps these five models on Liberty (spirit).
B_DATASET="spirit"

# (ablation, model, dataset)
JOBS=(
    # --- A: calibration size  (default model × 3 datasets) ---
    "A ${DEFAULT_MODEL} bgl"
    "A ${DEFAULT_MODEL} spirit"
    "A ${DEFAULT_MODEL} thunderbird"

    # --- B: head-k            (5 models × spirit) ---
    "B llama3_8b-attn   ${B_DATASET}"
    "B mistral_7b-attn  ${B_DATASET}"
    "B granite3_8b-attn ${B_DATASET}"
    "B qwen2-attn       ${B_DATASET}"
    "B gemma2_9b-attn   ${B_DATASET}"

    # --- C: instruction       (default model × 3 datasets) ---
    "C ${DEFAULT_MODEL} bgl"
    "C ${DEFAULT_MODEL} spirit"
    "C ${DEFAULT_MODEL} thunderbird"

    # --- D: selection method  (default model × 3 datasets) ---
    "D ${DEFAULT_MODEL} bgl"
    "D ${DEFAULT_MODEL} spirit"
    "D ${DEFAULT_MODEL} thunderbird"
)

JOB="${JOBS[$SLURM_ARRAY_TASK_ID]}"
read -r ABLATION MODEL DATASET <<< "${JOB}"

echo ""
echo "=========================================================="
echo " Task ${SLURM_ARRAY_TASK_ID}: ablation ${ABLATION}  |  ${MODEL}  |  ${DATASET}"
echo "=========================================================="

python -u run_ablations.py \
    --ablation "${ABLATION}" \
    --model "${MODEL}" \
    --dataset "${DATASET}"

echo ""
echo "Done. Result at result/search/${DATASET}/${MODEL}_ablation_${ABLATION}.json"
