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

    # Summary table with absolute and percent savings (Baseline - RL)
    summary_df = pd.DataFrame({"baseline": total_bl, "rl": total_rl})
    summary_df["abs_saving"] = summary_df["baseline"] - summary_df["rl"]
    summary_df["pct_saving"] = (summary_df["abs_saving"] / summary_df["baseline"]).replace([np.inf, -np.inf], np.nan)
    st.dataframe(
        summary_df.rename(index={"kwh": "Energy (kWh)", "cost_total": "Cost (USD)", "emissions_kg": "Emissions (kgCO2e)"}),
        use_container_width=True,
    )

    # Downloads for metrics and rollouts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download KPIs (CSV)",
            summary_df.reset_index().rename(columns={"index": "metric"}).to_csv(index=False).encode("utf-8"),
            file_name="evaluation_kpis.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download Rollout (CSV)",
            df_plot.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="evaluation_rollout.csv",
            mime="text/csv",
        )
    # Actions download button rendered after the explanation below
    # Expose results programmatically via session_state
    st.session_state["eval_kpis_summary"] = summary_df
    st.session_state["eval_rollout_df"] = df_plot

    st.subheader("Actions")
    # Explain the action meaning to users
    deltas = [float(x) for x in env_eval.cfg.action_deltas]
    st.markdown(
        """
        - Actions are hourly HVAC cooling setpoint adjustments (°C) relative to a neutral baseline.
        - Available deltas: {deltas}
        - Negative values: lower the setpoint (cool more). Positive values: raise the setpoint (cool less).
        - Baseline policy uses the action closest to 0.0 (no change).
        - In this surrogate model, lowering the setpoint tends to increase energy; raising it tends to reduce energy. The effect is stronger during weekday working hours (08:00–18:00).
        - Comfort penalty applies when the temperature is hot (≥ {t_hot:.0f}°C) and the action lowers the setpoint (negative delta). Penalty grows with the magnitude relative to ±{band:.1f}°C.
        """.format(
            deltas=deltas,
            t_hot=float(env_eval.cfg.comfort_temp_c),
            band=float(env_eval.cfg.comfort_band_c),
        )
    )
    actions_series = pd.Series(actions, index=env_eval.df.index[: len(actions)], name="setpoint_delta")
    st.line_chart(actions_series)
    st.download_button(
        "Download Actions (CSV)",
        actions_series.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="evaluation_actions.csv",
        mime="text/csv",
    )
    st.session_state["eval_actions_series"] = actions_series


if __name__ == "__main__":
    main()
