#!/bin/bash
# Run the recommended-minimum ablation set:
#   A. Calibration size       — Liberty (spirit)
#   B. Head-k                  — Liberty (spirit)
#   C. Instruction sensitivity — BGL + Liberty
#   D. Head selection method   — BGL + Liberty
#
# 6 sweep runs total, all on a single model (default: llama3_8b-attn).
# Override with: bash scripts/run_ablations.sh <model_name>
set -e

MODEL=${1:-llama3_8b-attn}

echo "===== Ablation A: calibration size  (spirit) ====="
python -u run_ablations.py --ablation A --model ${MODEL} --dataset spirit

echo ""
echo "===== Ablation B: head-k             (spirit) ====="
python -u run_ablations.py --ablation B --model ${MODEL} --dataset spirit

echo ""
echo "===== Ablation C: instruction        (bgl) ====="
python -u run_ablations.py --ablation C --model ${MODEL} --dataset bgl

echo ""
echo "===== Ablation C: instruction        (spirit) ====="
python -u run_ablations.py --ablation C --model ${MODEL} --dataset spirit

echo ""
echo "===== Ablation D: selection method   (bgl) ====="
python -u run_ablations.py --ablation D --model ${MODEL} --dataset bgl

echo ""
echo "===== Ablation D: selection method   (spirit) ====="
python -u run_ablations.py --ablation D --model ${MODEL} --dataset spirit

echo ""
echo "===== All ablations done. Results in ./result/search/<dataset>/${MODEL}_ablation_*.json ====="
