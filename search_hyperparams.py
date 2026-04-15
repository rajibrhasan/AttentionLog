import argparse
import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils import open_config, create_model
from detector.utils import process_attn, calc_attn_score
from select_head import find_pos_div_index, find_top_div_index

DEFAULT_INSTRUCTIONS = [
    "Say xxxxxx",
    "Repeat this normal log entry exactly",
    "Summarize this standard server log entry",
    "Classify whether this log line is normal or abnormal",
    "Analyze the following system log message",
]

DEFAULT_NUM_DATA_VALUES = [50, 100, 200, 500]

HEAD_METHODS = [
    ("pos_div", {"n": 1}),
    ("top_k", {"portion": 0.01}),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_pools(dataset_name, max_head_sel, n_cal, n_test_normal,
                    n_test_anomaly, seed, windowed=False, **dataset_paths):
    """Load data and split into 3 disjoint pools: head selection, calibration, test."""
    total_normal = max_head_sel + n_cal + n_test_normal
    total_anomaly = max_head_sel + n_cal + n_test_anomaly

    if windowed:
        from data.windowed import load_windowed_sampled
        # Load from train CSV for head selection + calibration
        head_and_cal_normal, head_and_cal_anomaly = load_windowed_sampled(
            dataset_paths["train_csv"], n_normal=max_head_sel + n_cal,
            n_anomaly=max_head_sel + n_cal, seed=seed)
        # Load from test CSV for evaluation
        test_normal, test_anomaly = load_windowed_sampled(
            dataset_paths["test_csv"], n_normal=n_test_normal,
            n_anomaly=n_test_anomaly, seed=seed)
        all_normal = head_and_cal_normal + test_normal
        all_anomaly = head_and_cal_anomaly + test_anomaly

        # Return early with proper splits (no re-shuffling needed)
        head_normal = head_and_cal_normal[:max_head_sel]
        cal_normal = head_and_cal_normal[max_head_sel:]
        head_anomaly = head_and_cal_anomaly[:max_head_sel]
        cal_anomaly = head_and_cal_anomaly[max_head_sel:]
        test_data = test_normal + test_anomaly
        random.seed(seed + 1)
        random.shuffle(test_data)
        return head_normal, head_anomaly, cal_normal, cal_anomaly, test_data

    elif dataset_name == "bgl":
        from data.bgl import load_bgl_sampled
        all_normal, all_anomaly = load_bgl_sampled(
            dataset_paths["bgl_path"], n_normal=total_normal,
            n_anomaly=total_anomaly, seed=seed)
    elif dataset_name == "hdfs":
        from data.hdfs import load_hdfs_sampled
        all_normal, all_anomaly = load_hdfs_sampled(
            dataset_paths["hdfs_log_path"], dataset_paths["hdfs_label_path"],
            n_normal=total_normal, n_anomaly=total_anomaly, seed=seed)
    elif dataset_name == "thunderbird":
        from data.thunderbird import load_thunderbird_sampled
        all_normal, all_anomaly = load_thunderbird_sampled(
            dataset_paths["thunderbird_path"], n_normal=total_normal,
            n_anomaly=total_anomaly, seed=seed)
    elif dataset_name == "spirit":
        from data.spirit import load_spirit_sampled
        all_normal, all_anomaly = load_spirit_sampled(
            dataset_paths["spirit_path"], n_normal=total_normal,
            n_anomaly=total_anomaly, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    random.seed(seed)
    random.shuffle(all_normal)
    random.shuffle(all_anomaly)

    head_normal = all_normal[:max_head_sel]
    cal_normal = all_normal[max_head_sel:max_head_sel + n_cal]
    test_normal = all_normal[max_head_sel + n_cal:
                             max_head_sel + n_cal + n_test_normal]

    head_anomaly = all_anomaly[:max_head_sel]
    cal_anomaly = all_anomaly[max_head_sel:max_head_sel + n_cal]
    test_anomaly = all_anomaly[max_head_sel + n_cal:
                               max_head_sel + n_cal + n_test_anomaly]

    test_data = test_normal + test_anomaly
    random.seed(seed + 1)
    random.shuffle(test_data)

    return head_normal, head_anomaly, cal_normal, cal_anomaly, test_data


def run_inference_batch(model, instruction, texts, desc="inference"):
    """Run inference and return processed heatmaps."""
    heatmaps = []
    for text in tqdm(texts, desc=desc):
        _, _, attention_maps, _, input_range, _ = model.inference(
            instruction, text, max_output_tokens=1)
        heatmap = process_attn(attention_maps[0], input_range, "normalize_sum")
        heatmaps.append(heatmap)
    return heatmaps


def compute_divergence(normal_heatmaps, anomaly_heatmaps):
    """Compute divergence maps between normal and anomaly attention patterns."""
    normal_arr = np.array(normal_heatmaps)
    anomaly_arr = np.array(anomaly_heatmaps)

    normal_mean = np.mean(normal_arr, axis=0)
    normal_std = np.std(normal_arr, axis=0)
    anomaly_mean = np.mean(anomaly_arr, axis=0)
    anomaly_std = np.std(anomaly_arr, axis=0)

    diff_map_mean = normal_mean - anomaly_mean
    diff_map_std = normal_std + anomaly_std

    return diff_map_mean, diff_map_std


def select_heads(diff_map_mean, diff_map_std, method_name, method_params):
    """Apply a head selection method. Returns list of [layer, head] pairs."""
    if method_name == "pos_div":
        return find_pos_div_index(diff_map_mean, diff_map_std, **method_params)
    elif method_name == "top_k":
        return find_top_div_index(diff_map_mean, diff_map_std, **method_params)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def evaluate_heads(heads, cal_normal_maps, cal_anomaly_maps,
                   test_maps, test_data):
    """Compute AUC from cached heatmaps for a given head set."""
    if not heads:
        return 0.5, False

    cal_pos_scores = [calc_attn_score(hm, heads) for hm in cal_normal_maps]
    cal_neg_scores = [calc_attn_score(hm, heads) for hm in cal_anomaly_maps]

    pos_mean = np.mean(cal_pos_scores)
    neg_mean = np.mean(cal_neg_scores)

    # Auto-detect flip direction (same logic as AttentionDetector)
    flip = neg_mean >= pos_mean

    labels = [d["label"] for d in test_data]
    test_scores_raw = [calc_attn_score(hm, heads) for hm in test_maps]

    if flip:
        scores = test_scores_raw
    else:
        scores = [1 - s for s in test_scores_raw]

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    return auc, flip


def search_hyperparams(model_name, dataset_name, instructions=None,
                       num_data_values=None, n_cal=200, n_test_normal=500,
                       n_test_anomaly=500, seed=42, output_path=None,
                       windowed=False, **dataset_paths):
    """
    Grid search over (instruction, num_data, head_method) to maximize AUC.

    Returns dict with 'best' result and 'all_results'.
    """
    set_seed(seed)

    if instructions is None:
        instructions = DEFAULT_INSTRUCTIONS
    if num_data_values is None:
        num_data_values = DEFAULT_NUM_DATA_VALUES

    max_head_sel = max(num_data_values)

    # Step 1: Load model once
    model_config_path = f"./configs/model_configs/{model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    model_config["params"]["max_output_tokens"] = 1
    model = create_model(config=model_config)
    model.print_model_info()

    # Step 2: Load data once (3-way split)
    print(f"\nLoading data for {dataset_name} (windowed={windowed})...")
    head_normal, head_anomaly, cal_normal, cal_anomaly, test_data = \
        load_data_pools(dataset_name, max_head_sel, n_cal,
                        n_test_normal, n_test_anomaly, seed,
                        windowed=windowed, **dataset_paths)

    head_normal_texts = [s["text"] for s in head_normal]
    head_anomaly_texts = [s["text"] for s in head_anomaly]
    cal_normal_texts = [s["text"] for s in cal_normal]
    cal_anomaly_texts = [s["text"] for s in cal_anomaly]
    test_texts = [d["text"] for d in test_data]

    print(f"Head selection pool: {len(head_normal_texts)} normal, "
          f"{len(head_anomaly_texts)} anomaly")
    print(f"Calibration pool: {len(cal_normal_texts)} normal, "
          f"{len(cal_anomaly_texts)} anomaly")
    print(f"Test pool: {len(test_data)} samples")

    # Step 3: Search
    all_results = []
    best_result = {"auc": -1}
    total_combos = len(instructions) * len(num_data_values) * len(HEAD_METHODS)
    combo_idx = 0

    print(f"\nTotal combinations: {total_combos}")
    print(f"  Instructions: {len(instructions)}")
    print(f"  num_data: {num_data_values}")
    print(f"  Head methods: {len(HEAD_METHODS)}")
    print("=" * 60)

    for instr_idx, instruction in enumerate(instructions):
        print(f"\n{'='*60}")
        print(f"[{instr_idx+1}/{len(instructions)}] Instruction: \"{instruction}\"")
        print(f"{'='*60}")

        # Run inference once per instruction on all pools
        print(f"  Inference: head selection pool ({max_head_sel}+{max_head_sel})...")
        head_normal_maps = run_inference_batch(
            model, instruction, head_normal_texts[:max_head_sel],
            desc="  head-sel normal")
        head_anomaly_maps = run_inference_batch(
            model, instruction, head_anomaly_texts[:max_head_sel],
            desc="  head-sel anomaly")

        print(f"  Inference: calibration pool ({n_cal}+{n_cal})...")
        cal_normal_maps = run_inference_batch(
            model, instruction, cal_normal_texts, desc="  cal normal")
        cal_anomaly_maps = run_inference_batch(
            model, instruction, cal_anomaly_texts, desc="  cal anomaly")

        print(f"  Inference: test pool ({len(test_data)})...")
        test_maps = run_inference_batch(
            model, instruction, test_texts, desc="  test")

        # For each num_data x head_method, compute from cached heatmaps
        for nd in num_data_values:
            nd_normal_maps = head_normal_maps[:nd]
            nd_anomaly_maps = head_anomaly_maps[:nd]

            diff_map_mean, diff_map_std = compute_divergence(
                nd_normal_maps, nd_anomaly_maps)

            ratio = diff_map_mean.max() / (diff_map_std.mean() + 1e-8)

            for method_name, method_params in HEAD_METHODS:
                combo_idx += 1
                heads = select_heads(diff_map_mean, diff_map_std,
                                     method_name, method_params)

                if not heads:
                    auc = 0.5
                    flip = False
                    print(f"  [{combo_idx}/{total_combos}] nd={nd}, "
                          f"{method_name}({method_params}): "
                          f"NO HEADS -> AUC=0.500")
                else:
                    auc, flip = evaluate_heads(
                        heads, cal_normal_maps, cal_anomaly_maps,
                        test_maps, test_data)
                    marker = " ***" if auc > best_result["auc"] else ""
                    print(f"  [{combo_idx}/{total_combos}] nd={nd}, "
                          f"{method_name}({method_params}): "
                          f"{len(heads)} heads, AUC={auc:.4f}, "
                          f"ratio={ratio:.3f}{marker}")

                result = {
                    "instruction": instruction,
                    "num_data": nd,
                    "head_method": method_name,
                    "head_params": method_params,
                    "num_heads": len(heads),
                    "heads": heads,
                    "auc": round(auc, 4),
                    "flip": flip,
                    "ratio": round(ratio, 4),
                }
                all_results.append(result)

                if auc > best_result["auc"]:
                    best_result = result

    # Step 4: Report
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best AUC: {best_result['auc']:.4f}")
    print(f"  Instruction: \"{best_result['instruction']}\"")
    print(f"  num_data: {best_result['num_data']}")
    print(f"  Head method: {best_result['head_method']}"
          f"({best_result.get('head_params', {})})")
    print(f"  Heads ({best_result['num_heads']}): {best_result['heads']}")
    print(f"  Flip: {best_result['flip']}")

    # Step 5: Save
    if output_path is None:
        output_path = f"./result/search/{dataset_name}/{model_name}_search.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "best": best_result,
        "all_results": sorted(all_results, key=lambda x: x["auc"],
                              reverse=True),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for attention-based anomaly detection")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["bgl", "hdfs", "thunderbird", "spirit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cal", type=int, default=200)
    parser.add_argument("--n_test_normal", type=int, default=500)
    parser.add_argument("--n_test_anomaly", type=int, default=500)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--bgl_path", type=str, default="./data/bgl/BGL.log")
    parser.add_argument("--hdfs_log_path", type=str, default="./data/hdfs/HDFS.log")
    parser.add_argument("--hdfs_label_path", type=str,
                        default="./data/hdfs/anomaly_label.csv")
    parser.add_argument("--thunderbird_path", type=str,
                        default="./data/thunderbird/Thunderbird.log")
    parser.add_argument("--spirit_path", type=str,
                        default="./data/spirit/spirit2.log")
    parser.add_argument("--windowed", action="store_true",
                        help="Use windowed sessions from preprocessed CSVs")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to windowed train.csv (used with --windowed)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to windowed test.csv (used with --windowed)")
    args = parser.parse_args()

    search_hyperparams(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        seed=args.seed,
        n_cal=args.n_cal,
        n_test_normal=args.n_test_normal,
        n_test_anomaly=args.n_test_anomaly,
        output_path=args.output_path,
        windowed=args.windowed,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        bgl_path=args.bgl_path,
        hdfs_log_path=args.hdfs_log_path,
        hdfs_label_path=args.hdfs_label_path,
        thunderbird_path=args.thunderbird_path,
        spirit_path=args.spirit_path,
    )
