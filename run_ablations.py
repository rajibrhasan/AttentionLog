"""Ablation driver for Attention-Log.

Usage:
    python run_ablations.py --ablation A --model llama3_8b-attn --dataset spirit
    python run_ablations.py --ablation B --model llama3_8b-attn --dataset spirit
    python run_ablations.py --ablation C --model llama3_8b-attn --dataset bgl
    python run_ablations.py --ablation D --model llama3_8b-attn --dataset bgl
"""
import argparse
import os
from search_hyperparams import search_hyperparams

# Default base instruction used by ablations that don't sweep instructions.
BASE_INSTRUCTION = "Summarize this standard server log entry"

ABLATIONS = {
    "A": {  # Calibration size sensitivity
        "instructions": [BASE_INSTRUCTION],
        "num_data_values": [10, 25, 50, 100, 200, 500],
        "head_methods": [("top_k", {"portion": 0.01})],
    },
    "B": {  # Head selection percentage k
        "instructions": [BASE_INSTRUCTION],
        "num_data_values": [50],
        "head_methods": [
            ("top_k", {"portion": 0.005}),
            ("top_k", {"portion": 0.01}),
            ("top_k", {"portion": 0.02}),
            ("top_k", {"portion": 0.05}),
            ("top_k", {"portion": 0.10}),
        ],
    },
    "C": {  # Instruction sensitivity
        "instructions": [
            "Say xxxxxx",
            "Repeat this normal log entry exactly",
            "Summarize this standard server log entry",
            "Classify whether this log line is normal or abnormal",
            "Analyze the following system log message",
        ],
        "num_data_values": [50],
        "head_methods": [("top_k", {"portion": 0.01})],
    },
    "D": {  # Head selection method
        "instructions": [BASE_INSTRUCTION],
        "num_data_values": [50],
        "head_methods": [
            ("pos_div", {"n": 1}),
            ("pos_div", {"n": 2}),
            ("pos_div", {"n": 3}),
            ("top_k", {"portion": 0.01}),
            ("top_k", {"portion": 0.05}),
        ],
    },
}

DATASET_PATHS = {
    "bgl_path": "./data/bgl/BGL.log",
    "spirit_path": "./data/spirit/spirit2_5m.log",
    "thunderbird_path": "./data/thunderbird/Thunderbird.log",
    "hdfs_log_path": "./data/hdfs/HDFS.log",
    "hdfs_label_path": "./data/hdfs/anomaly_label.csv",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=list(ABLATIONS.keys()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["bgl", "spirit", "thunderbird"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cal", type=int, default=200)
    parser.add_argument("--n_test_normal", type=int, default=500)
    parser.add_argument("--n_test_anomaly", type=int, default=500)
    args = parser.parse_args()

    cfg = ABLATIONS[args.ablation]
    out_dir = f"./result/search/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/{args.model}_ablation_{args.ablation}.json"

    print(f"\n{'='*60}")
    print(f"Ablation {args.ablation}  |  {args.model}  |  {args.dataset}")
    print(f"  instructions:    {len(cfg['instructions'])}")
    print(f"  num_data_values: {cfg['num_data_values']}")
    print(f"  head_methods:    {cfg['head_methods']}")
    print(f"  output:          {output_path}")
    print(f"{'='*60}\n")

    search_hyperparams(
        model_name=args.model,
        dataset_name=args.dataset,
        instructions=cfg["instructions"],
        num_data_values=cfg["num_data_values"],
        head_methods=cfg["head_methods"],
        seed=args.seed,
        n_cal=args.n_cal,
        n_test_normal=args.n_test_normal,
        n_test_anomaly=args.n_test_anomaly,
        output_path=output_path,
        **DATASET_PATHS,
    )


if __name__ == "__main__":
    main()
