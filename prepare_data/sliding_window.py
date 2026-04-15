import argparse
import os.path

import numpy as np
import pandas as pd
from helper import fixedSize_window, structure_log


LOG_FORMATS = {
    "bgl": "<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>",
    "thunderbird": "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>",
    "spirit": "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>",
    "liberty": "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>",
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess logs into windowed sessions")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the raw log file")
    parser.add_argument("--log_name", type=str, required=True,
                        help="Log filename (e.g., BGL.log, spirit2_5m.log)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["bgl", "thunderbird", "spirit", "liberty"],
                        help="Dataset type (determines log format)")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--start_line", type=int, default=0)
    parser.add_argument("--end_line", type=int, default=None)
    args = parser.parse_args()

    log_format = LOG_FORMATS[args.dataset]
    print(f"Dataset: {args.dataset}, log_format: {log_format}")

    output_dir = args.data_dir
    structure_log(args.data_dir, output_dir, args.log_name, log_format,
                  start_line=args.start_line, end_line=args.end_line)

    print(f"window_size: {args.window_size}; step_size: {args.step_size}")

    df = pd.read_csv(os.path.join(output_dir, f'{args.log_name}_structured.csv'))
    print(f"Total log lines: {len(df)}")

    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    train_len = int(args.train_ratio * len(df))
    df_train = df[:train_len]
    df_test = df[train_len:]
    df_test = df_test.reset_index(drop=True)

    print("Start grouping.")

    session_train_df = fixedSize_window(
        df_train[['Content', 'Label']],
        window_size=args.window_size, step_size=args.step_size
    )
    session_test_df = fixedSize_window(
        df_test[['Content', 'Label']],
        window_size=args.window_size, step_size=args.step_size
    )

    col = ['Content', 'Label', 'item_Label']
    spliter = ' ;-; '

    session_train_df = session_train_df[col]
    session_train_df['session_length'] = session_train_df["Content"].apply(len)
    session_train_df["Content"] = session_train_df["Content"].apply(lambda x: spliter.join(x))

    session_test_df = session_test_df[col]
    session_test_df['session_length'] = session_test_df["Content"].apply(len)
    session_test_df["Content"] = session_test_df["Content"].apply(lambda x: spliter.join(x))

    session_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    session_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"\nTrain: {len(session_train_df)} sessions "
          f"({int(session_train_df['Label'].sum())} anomaly, "
          f"{len(session_train_df) - int(session_train_df['Label'].sum())} normal)")
    print(f"Test: {len(session_test_df)} sessions "
          f"({int(session_test_df['Label'].sum())} anomaly, "
          f"{len(session_test_df) - int(session_test_df['Label'].sum())} normal)")
    print(f"\nSaved to {os.path.join(output_dir, 'train.csv')} and {os.path.join(output_dir, 'test.csv')}")
