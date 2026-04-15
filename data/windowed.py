import random
import pandas as pd


def load_windowed_sampled(csv_path, n_normal=500, n_anomaly=500, seed=42):
    """Load windowed sessions from a preprocessed CSV and sample balanced subsets.

    Args:
        csv_path: Path to train.csv or test.csv (from prepare_data scripts).
        n_normal: Number of normal sessions to sample.
        n_anomaly: Number of anomaly sessions to sample.
        seed: Random seed.

    Returns:
        (normal_samples, anomaly_samples) — each a list of dicts with
        'text' and 'label' keys.
    """
    df = pd.read_csv(csv_path)

    normal_df = df[df['Label'] == 0]
    anomaly_df = df[df['Label'] == 1]

    random.seed(seed)

    normal_indices = list(normal_df.index)
    anomaly_indices = list(anomaly_df.index)
    random.shuffle(normal_indices)
    random.shuffle(anomaly_indices)

    normal_samples = [
        {"text": df.loc[i, "Content"], "label": 0}
        for i in normal_indices[:n_normal]
    ]
    anomaly_samples = [
        {"text": df.loc[i, "Content"], "label": 1}
        for i in anomaly_indices[:n_anomaly]
    ]

    return normal_samples, anomaly_samples


def load_windowed_all(csv_path):
    """Load all windowed sessions from a CSV without sampling.

    Returns:
        list of dicts with 'text' and 'label' keys.
    """
    df = pd.read_csv(csv_path)
    return [{"text": row["Content"], "label": int(row["Label"])}
            for _, row in df.iterrows()]


def load_windowed_split(train_csv, test_csv, n_train_normal=200,
                        n_train_anomaly=200, n_test_normal=None,
                        n_test_anomaly=None, seed=42):
    """Load windowed sessions with train/test split.

    Train set is sampled for calibration.
    Test set uses ALL data by default (n_test_normal/n_test_anomaly=None).

    Returns:
        (train_normal, train_anomaly, test_data)
    """
    train_normal, train_anomaly = load_windowed_sampled(
        train_csv, n_normal=n_train_normal, n_anomaly=n_train_anomaly, seed=seed
    )

    if n_test_normal is None and n_test_anomaly is None:
        test_data = load_windowed_all(test_csv)
    else:
        n_test_normal = n_test_normal or 1000
        n_test_anomaly = n_test_anomaly or 1000
        test_normal, test_anomaly = load_windowed_sampled(
            test_csv, n_normal=n_test_normal, n_anomaly=n_test_anomaly, seed=seed
        )
        test_data = test_normal + test_anomaly

    random.seed(seed)
    random.shuffle(test_data)

    return train_normal, train_anomaly, test_data
