import os
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import get_building_ids, get_building_series, combine_meters_for_building
from src.utils import get_dataset_dir


@st.cache_data(show_spinner=False)
def _list_buildings(dataset_dir: Optional[str]):
    return get_building_ids(dataset_dir)


@st.cache_data(show_spinner=False)
def _load_building(building_id: str, dataset_dir: Optional[str]):
    df = combine_meters_for_building(building_id, dataset_dir, meters=("electricity", "gas", "steam", "chilledwater", "hotwater", "water", "solar"))
    return df


def main():
    st.title("📊 Data Explorer")

    dataset_dir = st.session_state.get("dataset_dir") or get_dataset_dir()
    st.caption(f"Dataset directory: {dataset_dir or 'Not found'}")
    if not dataset_dir or not os.path.isdir(dataset_dir):
        st.error("Dataset folder not found. Set it on the main page sidebar.")
        st.stop()

    with st.spinner("Loading building list..."):
        blds = _list_buildings(dataset_dir)
    if not blds:
        st.error("No building columns found in electricity.csv")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        b = st.selectbox("Select building", blds, index=0)
    with col2:
        meters_to_show = st.multiselect(
            "Meters",
            ["electricity", "gas", "steam", "chilledwater", "hotwater", "water", "solar"],
            default=["electricity"],
        )

    with st.spinner("Loading timeseries..."):
        df = _load_building(b, dataset_dir)
    if meters_to_show:
        df = df[[c for c in meters_to_show if c in df.columns]]
    # Guard against an empty selection (no matching meters)
    if df.shape[1] == 0:
        st.warning("No selected meters are available for this building. Choose another meter.")
        st.stop()

    df = df.asfreq("H")
    min_ts, max_ts = df.index.min(), df.index.max()
    start, end = st.slider("Date range", min_value=min_ts.to_pydatetime(), max_value=max_ts.to_pydatetime(), value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()))
    df_sel = df.loc[str(pd.to_datetime(start, utc=True)) : str(pd.to_datetime(end, utc=True))]

    st.subheader("Time Series")
    if df_sel.shape[1] > 0:
        fig = px.line(
            df_sel.reset_index(),
            x=df_sel.index.name or "timestamp",
            y=df_sel.columns,
            labels={"value": "Value", "index": "Time"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data in the selected date range.")

    st.subheader("Summary Stats")
    if df_sel.shape[1] > 0:
        st.dataframe(df_sel.describe().T, use_container_width=True)
    else:
        st.info("Nothing to summarise for the current selection.")


if __name__ == "__main__":
    main()
