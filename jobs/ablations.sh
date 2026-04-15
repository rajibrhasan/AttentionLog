#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# ==========================================================
# Ablation sweep — Attention-Log (sequential, single SLURM job)
#
#  A. Calibration size       — sweep DATASETS  (1 model × 3 datasets =  3 jobs)
#  B. Head-k                  — sweep MODELS    (5 models × 1 dataset =  5 jobs)
#  C. Instruction sensitivity — sweep DATASETS  (1 model × 3 datasets =  3 jobs)
#  D. Head selection method   — sweep DATASETS  (1 model × 3 datasets =  3 jobs)
#                                                 ─────────────────────────────
#                                                              total =  14 runs
#
# All 14 runs execute sequentially in this single SLURM job.
#
# Submit with:
#     sbatch jobs/ablations.sh
# ==========================================================

source ~/Attention-Tracker/.venv/bin/activate

echo "Node: $(hostname)"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

cd $HOME/NLPProject/AttentionLog

DEFAULT_MODEL="llama3_8b-attn"
B_DATASET="spirit"

# (ablation, model, dataset) — runs in order
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

TOTAL=${#JOBS[@]}
START_TIME=$(date +%s)

for i in "${!JOBS[@]}"; do
    JOB="${JOBS[$i]}"
    read -r ABLATION MODEL DATASET <<< "${JOB}"

    echo ""
    echo "=========================================================="
    echo " [$((i+1))/${TOTAL}] ablation ${ABLATION}  |  ${MODEL}  |  ${DATASET}"
    echo "=========================================================="

    python -u run_ablations.py \
        --ablation "${ABLATION}" \
        --model "${MODEL}" \
        --dataset "${DATASET}"

    echo "Result: result/search/${DATASET}/${MODEL}_ablation_${ABLATION}.json"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================================="
echo " All ${TOTAL} ablation runs complete in $((ELAPSED / 60)) min"
echo "=========================================================="
