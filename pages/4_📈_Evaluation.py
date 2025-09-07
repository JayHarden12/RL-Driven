import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.envs.building_energy_env import BuildingEnergyEnv, EnvConfig
from src.preprocess import compute_kpis, add_time_features, merge_weather


def rollout_env(env: BuildingEnergyEnv, policy, deterministic: bool = True) -> Tuple[pd.Series, List[float]]:
    obs, _ = env.reset()
    done = False
    kwh_values: List[float] = []
    actions: List[float] = []
    while not done:
        act = policy(obs, deterministic=deterministic)
        if isinstance(act, tuple):
            act = act[0]
        obs, reward, done, truncated, info = env.step(int(act))
        kwh_values.append(info.get("kwh", np.nan))
        actions.append(info.get("delta", 0.0))
        if done or truncated:
            break
    # Align to env df index
    idx = env.df.index[: len(kwh_values)]
    return pd.Series(kwh_values, index=idx, name="kwh"), actions


def main():
    st.title("📈 Evaluation")

    model = st.session_state.get("rl_model")
    feats_eval = st.session_state.get("rl_feats_eval")
    env_train = st.session_state.get("rl_env_train")

    if model is None or feats_eval is None or env_train is None:
        st.info("Train a model on the RL Training page first.")
        st.stop()

    # Build evaluation env with eval features
    env_eval = BuildingEnergyEnv(feats_eval.copy(), config=env_train.cfg)

    st.subheader("Policy vs Baseline")
    with st.spinner("Rolling out policy and baseline..."):
        # Policy rollout
        kwh_rl, actions = rollout_env(env_eval, lambda obs, deterministic=True: model.predict(obs, deterministic=deterministic))

        # Baseline rollout: always choose delta closest to 0
        zero_action = int(np.argmin([abs(a) for a in env_eval.cfg.action_deltas]))
        def baseline_policy(_obs, deterministic=True):
            return zero_action
        env_bl = BuildingEnergyEnv(feats_eval.copy(), config=env_train.cfg)
        kwh_bl, _ = rollout_env(env_bl, baseline_policy)

    df_plot = pd.concat([kwh_rl.rename("RL"), kwh_bl.rename("Baseline")], axis=1).dropna()
    fig = px.line(df_plot.reset_index(), x=df_plot.index.name or "timestamp", y=["RL", "Baseline"], labels={"value": "kWh"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPIs")
    kpis_rl = compute_kpis(kwh_rl)
    kpis_bl = compute_kpis(kwh_bl)
    total_rl = kpis_rl[["kwh", "cost_total", "emissions_kg"]].sum()
    total_bl = kpis_bl[["kwh", "cost_total", "emissions_kg"]].sum()
    delta = (total_bl - total_rl).rename({"kwh": "Energy Saved (kWh)", "cost_total": "Cost Saved (USD)", "emissions_kg": "Emissions Avoided (kgCO2e)"})
    st.write("Baseline totals:")
    st.json({k: float(v) for k, v in total_bl.items()})
    st.write("RL totals:")
    st.json({k: float(v) for k, v in total_rl.items()})
    st.write("Improvements:")
    st.json({k: float(v) for k, v in delta.items()})

    st.subheader("Actions")
    st.line_chart(pd.Series(actions, index=env_eval.df.index[: len(actions)], name="setpoint_delta"))


if __name__ == "__main__":
    main()

