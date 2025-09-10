import os
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

from src.envs.building_energy_env import BuildingEnergyEnv
from src.preprocess import add_time_features, merge_weather, compute_kpis
from src.data_loader import get_building_ids, get_building_series
from src.utils import get_dataset_dir


def _preset_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tz = "UTC"
    train = df.loc[str(pd.Timestamp("2016-01-01 00:00", tz=tz)) : str(pd.Timestamp("2016-12-31 23:00", tz=tz))]
    val = df.loc[str(pd.Timestamp("2017-01-01 00:00", tz=tz)) : str(pd.Timestamp("2017-06-30 23:00", tz=tz))]
    test = df.loc[str(pd.Timestamp("2017-07-01 00:00", tz=tz)) : str(pd.Timestamp("2017-12-31 23:00", tz=tz))]
    return train, val, test


def _env_features(env: BuildingEnergyEnv) -> np.ndarray:
    # Build observation features per timestamp with prev_delta set to 0 (myopic)
    hour_sin = env._hour_sin_arr.astype(np.float32)
    hour_cos = env._hour_cos_arr.astype(np.float32)
    is_wkend = env._is_weekend_arr.astype(np.float32)
    if env.cfg.normalize:
        base = (env._baseline_arr.astype(np.float32) / max(1.0, env.elec_scale)).astype(np.float32)
        temp = ((env._temp_arr.astype(np.float32) - env.temp_mean) / env.temp_std).astype(np.float32)
    else:
        base = env._baseline_arr.astype(np.float32)
        temp = env._temp_arr.astype(np.float32)
    prev_delta = np.zeros_like(base, dtype=np.float32)
    return np.stack([hour_sin, hour_cos, is_wkend, base, temp, prev_delta], axis=1)


def _best_action_labels(env: BuildingEnergyEnv) -> np.ndarray:
    # Compute myopic best action per timestep, matching the env's cost structure without noise
    actions = np.array(env.cfg.action_deltas, dtype=float)
    n = len(env._baseline_arr)
    labels = np.zeros(n, dtype=np.int64)
    for t in range(n):
        base_kwh = float(env._baseline_arr[t])
        is_weekend = env._is_weekend_arr[t] >= 0.5
        hour = int(env._hours_arr[t])
        working = 1.0 if (not is_weekend and 8 <= hour <= 18) else 0.5
        temp_c = float(env._temp_arr[t])
        costs = []
        for i, d in enumerate(actions):
            eff = 1.0 - env.cfg.beta_load_per_deg * (-d) * working
            eff = max(0.6, min(1.4, eff))
            kwh = max(0.0, base_kwh * eff)
            penalty = 0.0
            if temp_c >= env.cfg.comfort_temp_c and d < 0:
                penalty = (abs(d) / env.cfg.comfort_band_c) ** 2
            cost = env.cfg.lambda_energy * (kwh / max(1.0, env.elec_scale)) + env.cfg.lambda_comfort * penalty
            costs.append(cost)
        labels[t] = int(np.argmin(costs))
    return labels


def _rollout_with_classifier(env: BuildingEnergyEnv, pipe) -> pd.Series:
    # Use the classifier to choose actions given observations
    obs, _ = env.reset()
    done = False
    kwh_vals = []
    while not done:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        a = int(pipe.predict(x)[0])
        obs, r, done, truncated, info = env.step(a)
        kwh_vals.append(info.get("kwh", np.nan))
        if done or truncated:
            break
    idx = env.df.index[: len(kwh_vals)]
    return pd.Series(kwh_vals, index=idx, name="kwh")


def main():
    st.title("🧩 Non‑RL Baselines (RF / SVM / MLP)")

    dataset_dir = st.session_state.get("dataset_dir") or get_dataset_dir()
    if not dataset_dir or not os.path.isdir(dataset_dir):
        st.error("Dataset folder not found. Set it on the main page sidebar.")
        st.stop()

    # Building selection (reuse session selection if available)
    blds = get_building_ids(dataset_dir)
    default_b = 0
    if st.session_state.get("feature_building_id") in blds:
        default_b = blds.index(st.session_state["feature_building_id"])
    b = st.selectbox("Select building", blds, index=default_b)

    # Prepare features and preset splits
    with st.spinner("Loading and engineering features..."):
        s = (
            get_building_series(b, dataset_dir, source="electricity")
            .rename("electricity")
            .to_frame()
            .asfreq("H")
            .interpolate(limit_direction="both")
        )
        feats = merge_weather(add_time_features(s), dataset_dir, building_id=b)
        train_df, val_df, test_df = _preset_splits(feats)

    # Build train/test environments
    env_train = BuildingEnergyEnv(train_df.copy())
    env_test = BuildingEnergyEnv(test_df.copy(), config=env_train.cfg)

    X_train = _env_features(env_train)
    y_train = _best_action_labels(env_train)

    st.caption(f"Training samples: {len(y_train):,}. Test samples: {len(env_test.df):,}.")

    # Model choices
    st.subheader("Train Baseline Controllers")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rf = st.checkbox("RandomForest", True)
    with col2:
        use_svm = st.checkbox("SVM (RBF)", False)
    with col3:
        use_mlp = st.checkbox("MLP (Neural Net)", False)

    models: Dict[str, object] = {}
    if st.button("Train selected baselines"):
        with st.spinner("Training baselines..."):
            if use_rf:
                rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                models["RF"] = rf
            if use_svm:
                svm = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svc", SVC(C=1.0, kernel="rbf", gamma="scale")),
                ])
                svm.fit(X_train, y_train)
                models["SVM"] = svm
            if use_mlp:
                mlp = Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42)),
                ])
                mlp.fit(X_train, y_train)
                models["MLP"] = mlp
        st.success(f"Trained: {', '.join(models.keys()) or 'None'}")

        # Save
        os.makedirs("artifacts", exist_ok=True)
        for name, m in models.items():
            path = os.path.join("artifacts", f"{b}_NonRL_{name}.pkl")
            joblib.dump(m, path)
        st.caption("Saved models to artifacts/")

    # Evaluate any saved models on Test split
    st.subheader("Evaluate on Test (2017-H2)")
    found = []
    for name in ("RF", "SVM", "MLP"):
        p = os.path.join("artifacts", f"{b}_NonRL_{name}.pkl")
        if os.path.isfile(p):
            found.append((name, p))
    if not found:
        st.info("No saved baseline models found. Train and save above.")
        st.stop()

    selected = st.multiselect(
        "Baselines to compare",
        [n for n, _ in found],
        default=[n for n, _ in found],
    )
    if not selected:
        st.stop()

    series = {}
    for name, p in found:
        if name not in selected:
            continue
        pipe = joblib.load(p)
        s_kwh = _rollout_with_classifier(env_test, pipe)
        series[name] = s_kwh

    # Always include zero-change baseline for reference
    zero_action = int(np.argmin([abs(a) for a in env_test.cfg.action_deltas]))
    def _rollout_constant(env: BuildingEnergyEnv, action_idx: int) -> pd.Series:
        obs, _ = env.reset()
        done = False
        vals = []
        while not done:
            obs, r, done, truncated, info = env.step(action_idx)
            vals.append(info.get("kwh", np.nan))
            if done or truncated:
                break
        return pd.Series(vals, index=env.df.index[: len(vals)], name="kwh")
    kwh_bl = _rollout_constant(env_test, zero_action)

    df_plot = pd.concat([kwh_bl.rename("Baseline")] + [series[n].rename(n) for n in selected], axis=1).dropna()
    st.line_chart(df_plot)

    st.subheader("KPIs (vs Baseline)")
    from src.preprocess import compute_kpis
    k_bl = compute_kpis(kwh_bl)
    rows = []
    for name in selected:
        k = compute_kpis(series[name])
        tot = k[["kwh", "cost_total", "emissions_kg"]].sum()
        tot_bl = k_bl[["kwh", "cost_total", "emissions_kg"]].sum()
        rows.append({
            "model": name,
            "energy_kwh": float(tot["kwh"]),
            "cost_usd": float(tot["cost_total"]),
            "emissions_kg": float(tot["emissions_kg"]),
            "saving_kwh": float(tot_bl["kwh"] - tot["kwh"]),
            "saving_cost": float(tot_bl["cost_total"] - tot["cost_total"]),
            "saving_emissions": float(tot_bl["emissions_kg"] - tot["emissions_kg"]),
        })
    st.dataframe(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
