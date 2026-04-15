import random


def parse_bgl_line(line):
    """Parse a single BGL log line.

    BGL format: <Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>
    Label is "-" for normal, an alert category tag for anomaly.

    Returns:
        dict with 'label' (0=normal, 1=anomaly) and 'text' (log content),
        or None if the line cannot be parsed.
    """
    parts = line.strip().split(None, 9)
    if len(parts) < 10:
        return None
    label = 0 if parts[0] == "-" else 1
    content = parts[9]
    return {"label": label, "text": content}


def load_bgl_sampled(data_path, n_normal=500, n_anomaly=500, seed=42):
    """Load BGL data via reservoir sampling (single pass, memory-efficient).

    Args:
        data_path: Path to BGL.log file.
        n_normal: Number of normal samples to collect.
        n_anomaly: Number of anomaly samples to collect.
        seed: Random seed for reproducibility.

    Returns:
        (normal_samples, anomaly_samples) — each a list of dicts with
        'text' and 'label' keys.
    """
    random.seed(seed)
    normal_reservoir = []
    anomaly_reservoir = []
    normal_count = 0
    anomaly_count = 0

    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_bgl_line(line)
            if parsed is None:
                continue

            if parsed["label"] == 0:
                normal_count += 1
                if len(normal_reservoir) < n_normal:
                    normal_reservoir.append(parsed)
                else:
                    j = random.randint(0, normal_count - 1)
                    if j < n_normal:
                        normal_reservoir[j] = parsed
            else:
                anomaly_count += 1
                if len(anomaly_reservoir) < n_anomaly:
                    anomaly_reservoir.append(parsed)
                else:
                    j = random.randint(0, anomaly_count - 1)
                    if j < n_anomaly:
                        anomaly_reservoir[j] = parsed

    return normal_reservoir, anomaly_reservoir


def load_bgl_split(data_path, n_train_normal=100, n_train_anomaly=100,
                   n_test_normal=500, n_test_anomaly=500, seed=42):
    """Load BGL data with train/test split via reservoir sampling.

    Train set is used for head selection and threshold calibration.
    Test set is used for evaluation.

    Returns:
        (train_normal, train_anomaly, test_data)
        Each entry is a dict with 'text' and 'label' keys.
        test_data is shuffled.
    """
    total_normal = n_train_normal + n_test_normal
    total_anomaly = n_train_anomaly + n_test_anomaly

    normal_samples, anomaly_samples = load_bgl_sampled(
        data_path, n_normal=total_normal, n_anomaly=total_anomaly, seed=seed
    )

    random.seed(seed)
    random.shuffle(normal_samples)
    random.shuffle(anomaly_samples)

    train_normal = normal_samples[:n_train_normal]
    test_normal = normal_samples[n_train_normal:n_train_normal + n_test_normal]

    train_anomaly = anomaly_samples[:n_train_anomaly]
    test_anomaly = anomaly_samples[n_train_anomaly:n_train_anomaly + n_test_anomaly]

    test_data = test_normal + test_anomaly
    random.shuffle(test_data)

    return train_normal, train_anomaly, test_data
