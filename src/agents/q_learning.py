from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class QLearningConfig:
    n_bins: int = 8
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    seed: int | None = None


class TabularQLearner:
    """Tabular Q-learning over a discretized observation space.

    - Discretizes each observation dimension into `n_bins` (with sensible defaults
      for binary/[-1,1] features) using quantile-clipped ranges from a rollout.
    - Maintains a sparse Q-table as a dict: state_key -> action_values.
    - Provides a `predict(obs, deterministic=True)` method compatible with SB3 usage.
    """

    def __init__(self, env, config: QLearningConfig | None = None):
        self.env = env
        self.cfg = config or QLearningConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n_actions = int(env.action_space.n)
        # Discretization: list of bin edges arrays per dim
        self.bin_edges: List[np.ndarray] = []
        # Sparse Q-table mapping discretized state to action values
        self.Q: Dict[Tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        self._fitted = False

    # ------------------ Discretization utilities ------------------
    def _zero_action(self) -> int:
        # Choose the action whose delta is closest to 0
        if hasattr(self.env, "cfg") and hasattr(self.env.cfg, "action_deltas"):
            deltas = list(self.env.cfg.action_deltas)
            return int(np.argmin([abs(a) for a in deltas]))
        return 0

    def _collect_states(self, max_steps: int | None = None) -> np.ndarray:
        obs_list: List[np.ndarray] = []
        obs, _ = self.env.reset()
        done = False
        steps = 0
        zero_act = self._zero_action()
        while not done:
            obs_list.append(np.array(obs, dtype=np.float32))
            obs, _, done, truncated, _ = self.env.step(zero_act)
            steps += 1
            if truncated or done:
                break
            if max_steps is not None and steps >= max_steps:
                break
        return np.asarray(obs_list, dtype=np.float32)

    def _compute_bin_edges(self, X: np.ndarray) -> List[np.ndarray]:
        n_dims = X.shape[1]
        edges: List[np.ndarray] = []
        for i in range(n_dims):
            # Heuristics for known feature ranges
            if i in (0, 1):  # hour_sin, hour_cos in [-1, 1]
                e = np.linspace(-1.0, 1.0, max(2, self.cfg.n_bins) - 1, dtype=np.float32)
            elif i == 2:  # is_weekend in {0,1}
                e = np.array([0.5], dtype=np.float32)  # 2 bins: [0], [1]
            elif i == 5:  # prev_delta normalized ~ [-1, 1]
                e = np.linspace(-1.0, 1.0, max(2, self.cfg.n_bins) - 1, dtype=np.float32)
            else:
                low, high = np.percentile(X[:, i], [1, 99])
                if not np.isfinite(low) or not np.isfinite(high) or low == high:
                    low, high = float(np.min(X[:, i], initial=-1)), float(np.max(X[:, i], initial=1))
                    if low == high:
                        low, high = low - 1.0, high + 1.0
                e = np.linspace(low, high, max(2, self.cfg.n_bins) - 1, dtype=np.float32)
            # ensure strictly increasing
            e = np.unique(e)
            if e.size == 0:
                e = np.array([0.0], dtype=np.float32)
            edges.append(e)
        return edges

    def _discretize(self, obs: Sequence[float]) -> Tuple[int, ...]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        idxs: List[int] = []
        for i, e in enumerate(self.bin_edges):
            if i == 2:  # is_weekend
                # two bins: <=0.5 -> 0, >0.5 -> 1
                idx = 1 if obs_arr[i] > 0.5 else 0
            else:
                idx = int(np.digitize(obs_arr[i], e))
                # clip to valid bin index [0, n_bins]
                idx = int(np.clip(idx, 0, e.size))
            idxs.append(idx)
        return tuple(idxs)

    # ------------------ Learning ------------------
    def learn(self, total_timesteps: int = 10_000):
        # Fit discretizer from a rollout, then reset env for training
        sample_steps = min(total_timesteps, getattr(self.env, "df", None).shape[0] if hasattr(self.env, "df") else 2000)
        X = self._collect_states(max_steps=int(sample_steps))
        self.bin_edges = self._compute_bin_edges(X)
        self._fitted = True

        steps = 0
        epsilon = self.cfg.epsilon_start
        obs, _ = self.env.reset()
        done = False
        while steps < total_timesteps:
            s = self._discretize(obs)
            # Epsilon-greedy action selection
            if self.rng.random() < epsilon:
                a = int(self.rng.integers(self.n_actions))
            else:
                q_vals = self.Q[s]
                a = int(np.argmax(q_vals))

            next_obs, reward, done, truncated, _info = self.env.step(a)
            s_next = self._discretize(next_obs)

            # Q-learning update
            q_sa = self.Q[s][a]
            td_target = float(reward) + self.cfg.gamma * (np.max(self.Q[s_next]) if not (done or truncated) else 0.0)
            self.Q[s][a] = q_sa + self.cfg.alpha * (td_target - q_sa)

            obs = next_obs
            steps += 1
            # Epsilon decay
            if self.cfg.epsilon_decay_steps > 0:
                frac = min(1.0, steps / float(self.cfg.epsilon_decay_steps))
                epsilon = self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

            if done or truncated:
                obs, _ = self.env.reset()
                done = False
        return self

    # ------------------ Inference ------------------
    def predict(self, obs, deterministic: bool = True):
        if not self._fitted or not self.bin_edges:
            # Fallback: random/zero action
            a = 0 if deterministic else int(self.rng.integers(self.n_actions))
            return a, None
        s = self._discretize(obs)
        q_vals = self.Q[s]
        a = int(np.argmax(q_vals))
        return a, None

    # ------------------ Persistence ------------------
    def save(self, path: str):
        import pickle

        payload = {
            "cfg": self.cfg,
            "bin_edges": [np.asarray(e) for e in self.bin_edges],
            "Q": {k: np.asarray(v) for k, v in self.Q.items()},
            "n_actions": self.n_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, env, path: str) -> "TabularQLearner":
        import pickle

        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(env, payload.get("cfg", QLearningConfig()))
        obj.bin_edges = [np.asarray(e) for e in payload.get("bin_edges", [])]
        q: Dict[Tuple[int, ...], np.ndarray] = {}
        for k, v in payload.get("Q", {}).items():
            q[tuple(k) if not isinstance(k, tuple) else k] = np.asarray(v)
        obj.Q = defaultdict(lambda: np.zeros(obj.n_actions, dtype=np.float32))
        obj.Q.update(q)
        obj._fitted = True
        return obj

