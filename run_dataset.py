import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils import open_config, create_model
from detector.attn import AttentionDetector
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_log_split(args):
    """Load train/test split for a log anomaly dataset."""
    cal = args.n_cal_samples
    tn = args.n_test_normal
    ta = args.n_test_anomaly
    seed = args.seed

    if args.windowed:
        from data.windowed import load_windowed_split
        return load_windowed_split(args.train_csv, args.test_csv,
                                   cal, cal, tn, ta, seed=seed)
    elif args.dataset_name == "bgl":
        from data.bgl import load_bgl_split
        return load_bgl_split(args.bgl_path, cal, cal, tn, ta, seed)
    elif args.dataset_name == "hdfs":
        from data.hdfs import load_hdfs_split
        return load_hdfs_split(args.hdfs_log_path, args.hdfs_label_path,
                               cal, cal, tn, ta, seed=seed)
    elif args.dataset_name == "thunderbird":
        from data.thunderbird import load_thunderbird_split
        return load_thunderbird_split(args.thunderbird_path, cal, cal, tn, ta, seed)
    elif args.dataset_name == "spirit":
        from data.spirit import load_spirit_split
        return load_spirit_split(args.spirit_path, cal, cal, tn, ta, seed)


def main(args):
    set_seed(args.seed)

    output_logs = f"./result/{args.dataset_name}/{args.model_name}-{args.seed}.json"
    output_result = f"./result/{args.dataset_name}/result.jsonl"
    
    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)

    if args.heads:
        model_config["params"]["important_heads"] = json.loads(args.heads)

    model_config["params"]["max_output_tokens"] = 1
    model = create_model(config=model_config)
    model.print_model_info()

    log_datasets = {"bgl", "hdfs", "thunderbird", "spirit"}

    if args.dataset_name in log_datasets:
        train_normal, train_anomaly, test_data = _load_log_split(args)
        pos_examples = [s['text'] for s in train_normal]
        neg_examples = [s['text'] for s in train_anomaly]
        mode = "prefill" if args.prefill else "generate"
        detector = AttentionDetector(
            model, pos_examples=pos_examples, neg_examples=neg_examples,
            instruction=args.instruction,
            flip=True, mode=mode
        )
    else:
        dataset = load_dataset(args.dataset_name)
        test_data = dataset['test']
        detector = AttentionDetector(model)
    print("===================")
    print(f"Using detector: {detector.name}")

    labels, predictions, scores = [], [], []
    logs = []

    for data in tqdm(test_data):
        result = detector.detect(data['text'])
        detect = result[0]
        score = result[1]['focus_score']

        labels.append(data['label'])
        predictions.append(detect)
        if detector.flip:
            scores.append(score)  # higher score = more anomalous
        else:
            scores.append(1-score)  # lower score = more anomalous

        result_data = {
            "text": data['text'],
            "label": data['label'],
            "result": result
        }

        logs.append(result_data)

    auc_score = roc_auc_score(labels, scores)
    auprc_score = average_precision_score(labels, scores)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    auc_score = round(auc_score, 3)
    auprc_score = round(auprc_score, 3)
    fnr = round(fnr, 3)
    fpr = round(fpr, 3)
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)

    print(f"AUC: {auc_score}; AUPRC: {auprc_score}; Precision: {precision}; Recall: {recall}; F1: {f1}; FNR: {fnr}; FPR: {fpr}")
    
    os.makedirs(os.path.dirname(output_logs), exist_ok=True)
    with open(output_logs, "w") as f_out:
        f_out.write(json.dumps({"result": logs}, indent=4))

    os.makedirs(os.path.dirname(output_result), exist_ok=True)
    with open(output_result, "a") as f_out:
        f_out.write(json.dumps({
            "model": args.model_name,
            "seed": args.seed,
            "auc": auc_score,
            "auprc": auprc_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fnr": fnr,
            "fpr": fpr
        }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Script")
    
    parser.add_argument("--model_name", type=str, default="qwen2-attn",
                        help="Path to the model configuration file.")
    parser.add_argument("--dataset_name", type=str, default="deepset/prompt-injections", 
                        help="Path to the dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bgl_path", type=str, default="./data/bgl/BGL.log",
                        help="Path to BGL.log file")
    parser.add_argument("--hdfs_log_path", type=str, default="./data/hdfs/HDFS.log",
                        help="Path to HDFS.log file")
    parser.add_argument("--hdfs_label_path", type=str, default="./data/hdfs/anomaly_label.csv",
                        help="Path to HDFS anomaly_label.csv")
    parser.add_argument("--thunderbird_path", type=str, default="./data/thunderbird/Thunderbird.log",
                        help="Path to Thunderbird.log file")
    parser.add_argument("--spirit_path", type=str, default="./data/spirit/spirit2.log",
                        help="Path to Spirit log file")
    parser.add_argument("--n_test_normal", type=int, default=1000,
                        help="Number of normal test samples for log datasets")
    parser.add_argument("--n_test_anomaly", type=int, default=1000,
                        help="Number of anomaly test samples for log datasets")
    parser.add_argument("--n_cal_samples", type=int, default=200,
                        help="Number of calibration samples per class for threshold")
    parser.add_argument("--instruction", type=str, default="Repeat this normal log entry exactly",
                        help="System instruction for log anomaly detection")
    parser.add_argument("--windowed", action="store_true",
                        help="Use windowed sessions from preprocessed CSVs")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to windowed train.csv (used with --windowed)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Path to windowed test.csv (used with --windowed)")
    parser.add_argument("--heads", type=str, default=None,
                        help="JSON string of attention heads, e.g. '[[1,15],[5,4]]'. Overrides config file.")
    parser.add_argument("--prefill", action="store_true",
                        help="Use prefill-only attention (last data token → data tokens)")

    args = parser.parse_args()

    main(args)