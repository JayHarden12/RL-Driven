import os
import sys
from typing import Tuple

import pandas as pd


def detect_ts_col(path: str) -> str:
    df = pd.read_csv(path, nrows=5, engine="python")
    for c in ("timestamp", "time", "datetime", "date"):
        if c in df.columns:
            return c
    return df.columns[0]


def file_time_range(path: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    c = detect_ts_col(path)
    # Read only the timestamp col for speed and robustness
    s = pd.read_csv(
        path,
        usecols=[c],
        parse_dates=[c],
        engine="python",
        on_bad_lines="skip",
    )[c]
    s = pd.to_datetime(s, errors="coerce", utc=True).dropna()
    if s.empty:
        raise ValueError("no timestamps parsed")
    return s.min(), s.max()


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else r"Building Data Genome Project 2 dataset"
    paths = [
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.lower().endswith(".csv")
    ]
    for p in sorted(paths):
        name = os.path.basename(p)
        try:
            lo, hi = file_time_range(p)
            print(f"{name}: {lo} -> {hi}")
        except Exception as e:
            print(f"{name}: ERROR ({e})")


if __name__ == "__main__":
    main()

