import argparse
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from helper import structure_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess HDFS logs into session windows by Block ID")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the raw log file")
    parser.add_argument("--log_name", type=str, default="HDFS.log",
                        help="Log filename")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Path to anomaly_label.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    structure_log(args.data_dir, args.data_dir, args.log_name, log_format)

    spliter = ' ;-; '

    log_structured_file = os.path.join(args.data_dir, args.log_name + "_structured.csv")
    df = pd.read_csv(log_structured_file, engine='c',
                     na_filter=False, memory_map=True,
                     dtype={'Date': object, "Time": object})
    print(f"Number of messages in {log_structured_file}: {len(df)}")

    data_dict_content = defaultdict(list)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict_content[blk_Id].append(row["Content"])

    data_df = pd.DataFrame(list(data_dict_content.items()), columns=['BlockId', 'Content'])

    blk_label_dict = {}
    blk_df = pd.read_csv(args.label_file)
    for _, row in tqdm(blk_df.iterrows(), total=len(blk_df)):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x))

    train_len = int(args.train_ratio * len(data_df))
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    session_train_df = data_df[:train_len]
    session_test_df = data_df[train_len:].reset_index(drop=True)

    session_train_df['session_length'] = session_train_df["Content"].apply(len)
    session_train_df["Content"] = session_train_df["Content"].apply(lambda x: spliter.join(x))

    session_test_df['session_length'] = session_test_df["Content"].apply(len)
    session_test_df["Content"] = session_test_df["Content"].apply(lambda x: spliter.join(x))

    session_train_df.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    session_test_df.to_csv(os.path.join(args.data_dir, 'test.csv'), index=False)

    print(f"\nTrain: {len(session_train_df)} sessions "
          f"({int(session_train_df['Label'].sum())} anomaly, "
          f"{len(session_train_df) - int(session_train_df['Label'].sum())} normal)")
    print(f"Test: {len(session_test_df)} sessions "
          f"({int(session_test_df['Label'].sum())} anomaly, "
          f"{len(session_test_df) - int(session_test_df['Label'].sum())} normal)")
    print(f"\nSaved to {os.path.join(args.data_dir, 'train.csv')} and {os.path.join(args.data_dir, 'test.csv')}")
