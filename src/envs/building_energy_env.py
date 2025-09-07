from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    action_deltas: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)  # Cooling setpoint deltas [°C]
    beta_load_per_deg: float = 0.02  # Fractional load reduction per -1°C (heuristic)
    comfort_temp_c: float = 24.0
    comfort_band_c: float = 1.0
    lambda_energy: float = 1.0
    lambda_comfort: float = 0.5
    normalize: bool = True
    seed: Optional[int] = None


class BuildingEnergyEnv(gym.Env):
    """Simple surrogate HVAC setpoint control environment.

    - State: [hour_sin, hour_cos, is_weekend, baseline_norm, temp_norm, prev_delta]
    - Action: discrete setpoint delta in {action_deltas}
    - Reward: -(energy_kwh_weighted + comfort_penalty)
    """

    metadata = {"render_modes": []}

    def __init__(self, data: pd.DataFrame, config: EnvConfig = EnvConfig()):
        assert "electricity" in data.columns, "data must include 'electricity' column (kWh)"
        self.df = data.copy()
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        idx = self.df.index
        # Time features
        hours = idx.hour.values
        self.df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        self.df["is_weekend"] = ((idx.dayofweek >= 5).astype(float))

        # Baseline and temperature
        elec = self.df["electricity"].astype(float).clip(lower=0)
        self.baseline = elec.rolling(24, min_periods=1).mean()
        self.df["baseline"] = self.baseline
        temp_col = None
        for c in self.df.columns:
            if "temp" in c.lower():
                temp_col = c
                break
        if temp_col is None:
            # Synthesize a mild seasonal/diurnal temp if not provided
            t = np.arange(len(self.df))
            self.df["temp_c"] = 18 + 6 * np.sin(2 * np.pi * (t % 24) / 24)
            temp_col = "temp_c"
        self.temp_col = temp_col

        # Normalization scalers
        self.elec_scale = max(1.0, self.baseline.quantile(0.95))
        self.temp_mean = float(self.df[self.temp_col].mean())
        self.temp_std = float(self.df[self.temp_col].std() + 1e-6)

        self.action_space = gym.spaces.Discrete(len(self.cfg.action_deltas))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._t = 0
        self._prev_delta = 0.0

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self._t]
        base = float(row["baseline"]) / self.elec_scale if self.cfg.normalize else float(row["baseline"])
        temp = (float(row[self.temp_col]) - self.temp_mean) / self.temp_std if self.cfg.normalize else float(row[self.temp_col])
        obs = np.array([
            float(row["hour_sin"]),
            float(row["hour_cos"]),
            float(row["is_weekend"]),
            base,
            temp,
            float(self._prev_delta) / 2.0,
        ], dtype=np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._prev_delta = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        delta = float(self.cfg.action_deltas[int(action)])
        row = self.df.iloc[self._t]
        base_kwh = float(row["baseline"])  # Hourly baseline consumption

        # Load impact model: lower setpoint (negative delta) reduces load during working hours
        working = 1.0 if (row["is_weekend"] < 0.5 and 8 <= self.df.index[self._t].hour <= 18) else 0.5
        eff = 1.0 - self.cfg.beta_load_per_deg * (-delta) * working  # negative delta => reduce
        eff = max(0.6, min(1.4, eff))
        noise = self.rng.normal(0.0, 0.02 * base_kwh)
        kwh = max(0.0, base_kwh * eff + noise)

        # Comfort penalty: too low setpoint in hot conditions
        temp_c = float(row[self.temp_col])
        comfort_low = self.cfg.comfort_temp_c - self.cfg.comfort_band_c
        penalty = 0.0
        if temp_c >= self.cfg.comfort_temp_c and delta < 0:
            # Penalize aggressive cooling when hot outside (proxy for discomfort/overcooling)
            penalty = (abs(delta) / self.cfg.comfort_band_c) ** 2

        reward = -self.cfg.lambda_energy * (kwh / max(1.0, self.elec_scale)) - self.cfg.lambda_comfort * penalty

        self._t += 1
        self._prev_delta = delta
        terminated = self._t >= (len(self.df) - 1)
        info = {"kwh": kwh, "penalty": penalty, "delta": delta}
        return self._get_obs(), float(reward), terminated, False, info

