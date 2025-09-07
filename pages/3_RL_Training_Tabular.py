import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.envs.building_energy_env import BuildingEnergyEnv
from src.preprocess import add_time_features, merge_weather
from src.data_loader import get_building_series
from src.utils import get_dataset_dir


def _build_env_from_features(feats: pd.DataFrame) -> BuildingEnergyEnv:
    # Ensure required columns
    cols = list(feats.columns)
    if "electricity" not in cols:
        # If features came from raw loader, ensure electricity is present
        raise ValueError("Features must include 'electricity' column")
    df_env = feats.copy()
    return BuildingEnergyEnv(df_env)


def main():
    st.title("RL Training (Tabular + SB3)")

    dataset_dir = st.session_state.get("dataset_dir") or get_dataset_dir()
    if dataset_dir is None:
        st.error("Dataset path not set. Go to the main page to configure.")
        st.stop()

    # Obtain features from Feature Engineering page if present
    feats: Optional[pd.DataFrame] = st.session_state.get("features_df")
    b: Optional[str] = st.session_state.get("feature_building_id")

    if feats is None or b is None:
        st.info("No engineered features in session. Building minimal features now.")
        # Build fallback features
        from src.data_loader import get_building_ids

        bld_list = get_building_ids(dataset_dir)
        b = st.selectbox("Select building", bld_list, index=0)
        s = (
            get_building_series(b, dataset_dir)
            .rename("electricity")
            .to_frame()
            .asfreq("H")
            .interpolate(limit_direction="both")
        )
        feats = merge_weather(add_time_features(s), dataset_dir)

    assert feats is not None and b is not None
    st.caption(f"Using building: {b}")

    # Train/val split selection
    min_ts, max_ts = feats.index.min(), feats.index.max()
    default_start = min_ts.to_pydatetime().date()
    default_split = (min_ts.to_pydatetime() + (max_ts - min_ts) * 0.7).date()
    start_date = st.date_input("Train start", value=default_start)
    split_date = st.date_input("Validation start (end of training)", value=default_split)

    feats_train = feats.loc[str(pd.to_datetime(start_date, utc=True)) : str(pd.to_datetime(split_date, utc=True))]
    feats_eval = feats.loc[str(pd.to_datetime(split_date, utc=True)) :]

    env = _build_env_from_features(feats_train)

    algo = st.selectbox("Algorithm", ["PPO", "DQN", "Q-Learning (tabular)"], index=0)
    total_timesteps = st.slider(
        "Training steps", min_value=1_000, max_value=200_000, value=20_000, step=1_000
    )

    model = None
    algo_key = ""
    if algo in ("PPO", "DQN"):
        try:
            if algo == "PPO":
                from stable_baselines3 import PPO

                model = PPO("MlpPolicy", env, verbose=0)
                algo_key = "PPO"
            else:
                from stable_baselines3 import DQN

                model = DQN("MlpPolicy", env, verbose=0)
                algo_key = "DQN"
        except Exception as e:
            st.error(
                "Could not import Stable-Baselines3. Install with 'pip install -r requirements.txt'.\n" + str(e)
            )
            st.stop()
    else:
        from src.agents.q_learning import TabularQLearner, QLearningConfig

        st.markdown("Configure Q-Learning hyperparameters:")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_bins = st.slider("Bins per feature", 3, 20, 8, 1)
            alpha = st.slider("Alpha (LR)", 0.01, 1.0, 0.1, 0.01)
        with c2:
            gamma = st.slider("Gamma (discount)", 0.50, 0.999, 0.95, 0.001)
            eps_start = st.slider("Epsilon start", 0.0, 1.0, 1.0, 0.05)
        with c3:
            eps_end = st.slider("Epsilon end", 0.0, 1.0, 0.05, 0.01)
            eps_decay = st.slider(
                "Epsilon decay steps", 1_000, 200_000, min(total_timesteps, 10_000), 1_000
            )

        cfg = QLearningConfig(
            n_bins=int(n_bins),
            alpha=float(alpha),
            gamma=float(gamma),
            epsilon_start=float(eps_start),
            epsilon_end=float(eps_end),
            epsilon_decay_steps=int(eps_decay),
        )
        model = TabularQLearner(env, cfg)
        algo_key = "Q-Learning"

    if st.button("Train model"):
        with st.spinner("Training RL agent..."):
            model.learn(total_timesteps=total_timesteps)
        st.success("Training complete.")
        st.session_state["rl_model"] = model
        st.session_state["rl_env_train"] = env
        st.session_state["rl_feats_eval"] = feats_eval
        # Save model
        os.makedirs("artifacts", exist_ok=True)
        if algo_key in ("PPO", "DQN"):
            save_path = os.path.join("artifacts", f"{b}_{algo_key}_model.zip")
            model.save(save_path)
        else:
            save_path = os.path.join("artifacts", f"{b}_{algo_key}.pkl")
            model.save(save_path)
        st.caption(f"Saved model: {save_path}")


if __name__ == "__main__":
    main()
