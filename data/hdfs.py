import re
import csv
import random
from collections import defaultdict


BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")


def parse_hdfs_line(line):
    """Parse a single HDFS log line and extract block_id.

    HDFS format: <Date> <Time> <PID> <Level> <Component>: <Content>
    Block IDs are embedded in the content (e.g., blk_38865049064139660).

    Returns:
        dict with 'block_id' and 'text' (log content after component),
        or None if no block_id found or line cannot be parsed.
    """
    line = line.strip()
    if not line:
        return None

    match = BLOCK_ID_PATTERN.search(line)
    if not match:
        return None

    block_id = match.group(1)

    # Extract content: everything after "Level Component: "
    parts = line.split(None, 4)
    if len(parts) < 5:
        return None
    content = parts[4]

    return {"block_id": block_id, "text": content}


def load_hdfs_labels(label_path):
    """Load anomaly labels from anomaly_label.csv.

    CSV format: BlockId,Label
    Label values: 'Normal' or 'Anomaly'

    Returns:
        dict mapping block_id (str) -> label (int: 0=normal, 1=anomaly)
    """
    labels = {}
    with open(label_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_id = row["BlockId"]
            labels[block_id] = 0 if row["Label"] == "Normal" else 1
    return labels


def load_hdfs_traces(log_path, label_path, max_lines_per_trace=20):
    """Group HDFS log lines by block_id into traces with labels.

    Each trace concatenates the log content for a single block_id.
    Traces are truncated to max_lines_per_trace to keep input length manageable.

    Args:
        log_path: Path to HDFS.log file.
        label_path: Path to anomaly_label.csv file.
        max_lines_per_trace: Max log lines per trace to keep.

    Returns:
        List of dicts with 'text' (concatenated log content) and
        'label' (0=normal, 1=anomaly).
    """
    labels = load_hdfs_labels(label_path)

    # Group log lines by block_id
    traces = defaultdict(list)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_hdfs_line(line)
            if parsed is None:
                continue
            bid = parsed["block_id"]
            if bid in labels and len(traces[bid]) < max_lines_per_trace:
                traces[bid].append(parsed["text"])

    # Build trace entries
    data = []
    for bid, lines in traces.items():
        if bid not in labels:
            continue
        data.append({
            "text": " ".join(lines),
            "label": labels[bid]
        })

    return data


def load_hdfs_sampled(log_path, label_path, n_normal=500, n_anomaly=500,
                      max_lines_per_trace=20, seed=42):
    """Load HDFS traces and sample a balanced subset.

    Args:
        log_path: Path to HDFS.log file.
        label_path: Path to anomaly_label.csv file.
        n_normal: Number of normal traces to sample.
        n_anomaly: Number of anomaly traces to sample.
        max_lines_per_trace: Max log lines per trace.
        seed: Random seed.

    Returns:
        (normal_samples, anomaly_samples) — each a list of dicts with
        'text' and 'label' keys.
    """
    all_traces = load_hdfs_traces(log_path, label_path, max_lines_per_trace)

    normal = [t for t in all_traces if t["label"] == 0]
    anomaly = [t for t in all_traces if t["label"] == 1]

    random.seed(seed)
    random.shuffle(normal)
    random.shuffle(anomaly)

    return normal[:n_normal], anomaly[:n_anomaly]


def load_hdfs_split(log_path, label_path, n_train_normal=100,
                    n_train_anomaly=100, n_test_normal=500,
                    n_test_anomaly=500, max_lines_per_trace=20, seed=42):
    """Load HDFS data with train/test split.

    Returns:
        (train_normal, train_anomaly, test_data)
        Each entry is a dict with 'text' and 'label' keys.
        test_data is shuffled.
    """
    total_normal = n_train_normal + n_test_normal
    total_anomaly = n_train_anomaly + n_test_anomaly

    normal, anomaly = load_hdfs_sampled(
        log_path, label_path,
        n_normal=total_normal, n_anomaly=total_anomaly,
        max_lines_per_trace=max_lines_per_trace, seed=seed
    )

    random.seed(seed)

    train_normal = normal[:n_train_normal]
    test_normal = normal[n_train_normal:n_train_normal + n_test_normal]

    train_anomaly = anomaly[:n_train_anomaly]
    test_anomaly = anomaly[n_train_anomaly:n_train_anomaly + n_test_anomaly]

    test_data = test_normal + test_anomaly
    random.shuffle(test_data)

    return train_normal, train_anomaly, test_data
