from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd

from .utils import get_dataset_dir


TIMESTAMP_CANDIDATES = ["timestamp", "time", "datetime", "date"]


def _timestamp_col(df: pd.DataFrame) -> Optional[str]:
    for c in TIMESTAMP_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = _timestamp_col(df)
    if ts_col is None:
        # assume first column is timestamp
        ts_col = df.columns[0]
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    out = out.dropna(subset=[ts_col])
    out = out.sort_values(ts_col)
    out = out.set_index(ts_col)
    return out


def _read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV with sane fallbacks for occasional malformed rows.

    - Try the default C engine first (fastest).
    - On ParserError, retry with the Python engine (more forgiving).
    - If it still fails, skip bad lines with the Python engine.
    """
    try:
        return pd.read_csv(path)
    except Exception as e1:  # pandas.errors.ParserError et al.
        try:
            return pd.read_csv(path, engine="python")
        except Exception as e2:
            # Last resort: skip malformed lines; better to load most data than fail.
            return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _load_csv(name: str, dataset_dir: Optional[str] = None) -> pd.DataFrame:
    ds_dir = get_dataset_dir(dataset_dir)
    if not ds_dir:
        raise FileNotFoundError("Dataset directory not found. Set BDG2_DATASET_DIR or fix path in sidebar.")
    path = os.path.join(ds_dir, f"{name}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = _read_csv_robust(path)
    return df


def load_metadata(dataset_dir: Optional[str] = None) -> pd.DataFrame:
    return _load_csv("metadata", dataset_dir)


def load_meter(name: str = "electricity", dataset_dir: Optional[str] = None) -> pd.DataFrame:
    """Load a meter CSV (e.g., electricity, gas, chilledwater) as time-indexed wide DataFrame."""
    df = _load_csv(name, dataset_dir)
    df = _parse_times(df)
    # Some BDG2 CSVs may include an unnamed index column; drop if duplicated
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # type: ignore
    return df


def load_weather(dataset_dir: Optional[str] = None) -> pd.DataFrame:
    df = _load_csv("weather", dataset_dir)
    df = _parse_times(df)
    return df


def get_building_ids(dataset_dir: Optional[str] = None, source: str = "electricity") -> List[str]:
    df = load_meter(source, dataset_dir)
    # Assume non-time columns are buildings; since time is index already
    bcols = [c for c in df.columns if c.lower() not in ("timestamp", "time", "datetime", "date")]
    return bcols


def get_building_series(
    building_id: str,
    dataset_dir: Optional[str] = None,
    source: str = "electricity",
) -> pd.Series:
    df = load_meter(source, dataset_dir)
    if building_id not in df.columns:
        raise KeyError(f"Building '{building_id}' not found in {source}. Available: {len(df.columns)} columns")
    s = df[building_id].astype(float)
    s.name = source
    s = s.asfreq("H")  # BDG2 is hourly; ensure regularity
    return s


def combine_meters_for_building(
    building_id: str,
    dataset_dir: Optional[str] = None,
    meters: Tuple[str, ...] = ("electricity",),
) -> pd.DataFrame:
    frames = []
    for m in meters:
        try:
            s = get_building_series(building_id, dataset_dir, source=m)
            frames.append(s.rename(m))
        except (FileNotFoundError, KeyError):
            continue
    if not frames:
        raise FileNotFoundError(f"No meters found for building {building_id} in {meters}")
    out = pd.concat(frames, axis=1)
    return out
