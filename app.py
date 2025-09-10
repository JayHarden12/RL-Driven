import os
import streamlit as st

from src.utils import get_dataset_dir
from src.data_fetch import ensure_dataset


st.set_page_config(page_title="RL-Driven Building Energy Optimisation", layout="wide")


def init_dataset():
    """Silently ensure dataset exists on first run using secrets/env URL.

    Removes sidebar configuration; relies on auto-download and utils.find_dataset_dir.
    """
    current_path = st.session_state.get("dataset_dir") or get_dataset_dir()
    if (not current_path or not os.path.isdir(current_path)) and not st.session_state.get("dataset_autodownloaded", False):
        auto_url = st.secrets.get("BDG2_DATASET_URL", os.getenv("BDG2_DATASET_URL", ""))
        auto_sha = st.secrets.get("BDG2_DATASET_SHA256", os.getenv("BDG2_DATASET_SHA256", ""))
        if auto_url:
            try:
                path = ensure_dataset(auto_url, sha256=(auto_sha or None))
                if path and os.path.isdir(path):
                    st.session_state["dataset_dir"] = path
                    st.session_state["dataset_autodownloaded"] = True
            except Exception:
                # Fail silently; pages needing data will surface their own errors
                pass


def main():
    init_dataset()
    st.title("Reinforcement-Learning Driven Building Energy Optimisation")
    st.write(
        "Explore BDG2 data, engineer features, train a simple RL policy, and evaluate potential energy savings."
    )
    st.markdown(
        """
        - Use the sidebar to confirm the dataset path.
        - Navigate pages from the sidebar: Data Explorer, Feature Engineering, RL Training, and Evaluation.
        - The RL environment is a didactic surrogate for setpoint control; refine it to suit your building.
        """
    )

    st.subheader("Quick Links")
    # Keep link to Q-learning training page; classic RL Training page has been removed
    st.page_link("pages/3_RL_Training_Tabular.py", label="RL Training (Tabular + SB3)")


if __name__ == "__main__":
    main()
