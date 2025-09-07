import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.data_loader import get_building_ids, get_building_series
from src.preprocess import add_time_features, merge_weather
from src.utils import get_dataset_dir


@st.cache_data(show_spinner=False)
def _list_buildings(dataset_dir: Optional[str]):
    return get_building_ids(dataset_dir)


@st.cache_data(show_spinner=False)
def _load_series(building_id: str, dataset_dir: Optional[str]):
    return get_building_series(building_id, dataset_dir, source="electricity").rename("electricity").to_frame()


def main():
    st.title("🧪 Feature Engineering")
    dataset_dir = st.session_state.get("dataset_dir") or get_dataset_dir()
    if not dataset_dir or not os.path.isdir(dataset_dir):
        st.error("Dataset folder not found. Set it on the main page sidebar.")
        st.stop()

    blds = _list_buildings(dataset_dir)
    b = st.selectbox("Select building", blds, index=0)

    with st.spinner("Loading electricity series..."):
        df = _load_series(b, dataset_dir)
    df = df.asfreq("H").interpolate(limit_direction="both")

    st.caption("Adding time features and merging weather (if available)...")
    feats = add_time_features(df)
    feats = merge_weather(feats, dataset_dir)

    st.subheader("Preview")
    st.dataframe(feats.head(48), use_container_width=True)

    train_model = st.checkbox("Fit baseline consumption model (RandomForest)", value=False)
    model = None
    if train_model:
        target = "electricity"
        feature_cols = [c for c in feats.columns if c != target]
        X = feats[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
        y = feats[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        st.success(f"Baseline model trained. Holdout R^2 = {r2:.3f}")
        preds = pd.Series(model.predict(X), index=feats.index, name="baseline_model")
        st.line_chart(pd.concat([feats[target].rename("actual"), preds], axis=1))

    st.session_state["features_df"] = feats
    st.session_state["feature_building_id"] = b
    if model is not None:
        st.session_state["baseline_model_trained"] = True
    else:
        st.session_state["baseline_model_trained"] = False

    # Option to persist engineered features
    artifacts_dir = os.path.join("artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    if st.button("Save engineered features to artifacts/feats_parquet"):
        out_path = os.path.join(artifacts_dir, f"{b}_features.parquet")
        feats.to_parquet(out_path)
        st.success(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

