from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_weather, load_metadata


@dataclass
class Pricing:
    energy_price_per_kwh: float = 0.12  # USD/kWh
    carbon_intensity_kg_per_kwh: float = 0.35  # kgCO2e/kWh (example grid average)
    demand_charge_per_kw: float = 12.0  # USD/kW (monthly peak) — illustrative


def add_time_features(df: pd.DataFrame, index_col: Optional[str] = None) -> pd.DataFrame:
    if index_col is not None and index_col in df.columns:
        ts = pd.to_datetime(df[index_col], utc=True)
        df = df.set_index(ts)
    out = df.copy()
    idx = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_working_hour"] = ((out["hour"] >= 8) & (out["hour"] <= 18) & (out["is_weekend"] == 0)).astype(int)
    return out


def merge_weather(
    df: pd.DataFrame,
    dataset_dir: Optional[str] = None,
    building_id: Optional[str] = None,
) -> pd.DataFrame:
    """Join weather onto a single-building dataframe without exploding rows.

    If `building_id` is provided and metadata is available, use that building's
    `site_id` to select the corresponding weather series. Otherwise, aggregate
    weather across sites per timestamp to ensure a unique time index.
    """
    try:
        w = load_weather(dataset_dir)
    except FileNotFoundError:
        return df

    # Filter by site if possible to avoid duplicate timestamp rows
    w2 = w.copy()
    site = None
    if building_id is not None:
        try:
            meta = load_metadata(dataset_dir)
            row = meta.loc[meta["building_id"] == building_id]
            if not row.empty and "site_id" in row.columns:
                site = str(row.iloc[0]["site_id"]).strip()
        except Exception:
            site = None

    if site and "site_id" in w2.columns:
        w2 = w2[w2["site_id"].astype(str).str.strip() == site]
    # Drop identifier and ensure unique timestamp index
    w2 = w2.drop(columns=["site_id"], errors="ignore")
    if not w2.index.is_unique:
        w2 = w2[~w2.index.duplicated(keep="first")]

    # Heuristic: pick a temperature-like column
    temp_cols = [c for c in w2.columns if "temp" in c.lower() or c.lower() in ("t2m", "airtemperature")]
    use_cols = temp_cols[:1] if temp_cols else []
    feat = w2[use_cols].rename(columns=lambda c: c.lower()) if use_cols else pd.DataFrame(index=w2.index)

    out = df.join(feat, how="left").interpolate(limit_direction="both")
    return out


def aggregate(df: pd.DataFrame, rule: str = "H", cols: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    cols = cols or tuple(df.select_dtypes(include=[np.number]).columns)
    out = df[cols].resample(rule).mean()
    return out


def compute_kpis(consumption_kwh: pd.Series, pricing: Pricing = Pricing()) -> pd.DataFrame:
    """Compute hourly KPIs and allocate monthly demand charges evenly across hours.

    - Energy cost: kWh * price
    - Demand charge: monthly peak (kW ~= kWh for 1h) * demand rate, spread equally over hours in that month
    - Emissions: kWh * carbon intensity
    """
    s = consumption_kwh.fillna(0.0).clip(lower=0.0)
    df = pd.DataFrame({"kwh": s})

    # Energy cost per hour
    df["cost_energy"] = df["kwh"] * pricing.energy_price_per_kwh

    # Monthly peak kW from hourly series (resampled to Month Start)
    monthly_peak_kw = df["kwh"].resample("MS").max()
    monthly_demand_cost = monthly_peak_kw * pricing.demand_charge_per_kw

    # Allocate each month's demand cost evenly across all hours in that month
    periods = df.index.to_period("M")
    hours_per_month = periods.value_counts().sort_index()  # PeriodIndex -> counts
    # Map each timestamp to its month start (MS) timestamp key present in monthly_demand_cost
    # Convert PeriodIndex to Timestamp at month start in a version-compatible way
    month_start_idx = periods.to_timestamp(how="start")
    # Align monthly demand cost with each hourly timestamp via month start key
    demand_cost_per_ts = monthly_demand_cost.reindex(month_start_idx).to_numpy()
    # Divide by number of hours in the corresponding month
    hours_count_per_ts = periods.map(hours_per_month).to_numpy()
    df["cost_demand"] = np.divide(
        demand_cost_per_ts,
        np.maximum(1, hours_count_per_ts),
        out=np.zeros_like(demand_cost_per_ts, dtype=float),
        where=hours_count_per_ts > 0,
    )

    # Totals and emissions
    df["cost_total"] = df["cost_energy"] + df["cost_demand"]
    df["emissions_kg"] = df["kwh"] * pricing.carbon_intensity_kg_per_kwh
    return df
