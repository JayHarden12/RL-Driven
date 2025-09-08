import os
import streamlit as st

from src.utils import get_dataset_dir
from src.data_fetch import ensure_dataset


st.set_page_config(page_title="RL-Driven Building Energy Optimisation", layout="wide")


def sidebar():
    st.sidebar.title("Configuration")

    # Auto-download on first run if dataset is missing and URL is configured
    current_path = st.session_state.get("dataset_dir") or get_dataset_dir()
    if (not current_path or not os.path.isdir(current_path)) and not st.session_state.get("dataset_autodownloaded", False):
        auto_url = st.secrets.get("BDG2_DATASET_URL", os.getenv("BDG2_DATASET_URL", ""))
        auto_sha = st.secrets.get("BDG2_DATASET_SHA256", os.getenv("BDG2_DATASET_SHA256", ""))
        if auto_url:
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()

            def _progress(done: int, total: int | None):
                if total:
                    try:
                        prog.progress(min(1.0, done / total))
                    except Exception:
                        pass

            def _log(msg: str):
                status.info(msg)

            try:
                with st.sidebar.spinner("Downloading dataset (first run)..."):
                    path = ensure_dataset(auto_url, sha256=(auto_sha or None), progress=_progress, logger=_log)
                if path and os.path.isdir(path):
                    st.session_state["dataset_dir"] = path
                    st.session_state["dataset_autodownloaded"] = True
                    current_path = path
                    st.sidebar.success(f"Dataset ready at {path}")
                else:
                    st.sidebar.error("Dataset download finished but dataset folder not found. Check archive layout.")
            except Exception as e:
                st.sidebar.error(f"Auto-download failed: {e}")

    default_dir = current_path or os.path.abspath(".")
    dataset_dir = st.sidebar.text_input("Dataset folder", value=default_dir)
    if st.sidebar.button("Save dataset path"):
        st.session_state["dataset_dir"] = dataset_dir
        st.sidebar.success("Saved. Pages will use this path.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Set env var BDG2_DATASET_DIR to persist.")

    # Optional: Manual download control
    with st.sidebar.expander("Dataset download (manual)"):
        url_secret = st.secrets.get("BDG2_DATASET_URL", os.getenv("BDG2_DATASET_URL", ""))
        url = st.text_input(
            "Dataset URL",
            value=url_secret,
            help="Direct download URL to the BDG2 zip (e.g., GitHub Release/S3)",
        )
        sha = st.text_input("SHA256 (optional)", value=os.getenv("BDG2_DATASET_SHA256", ""))
        if st.button("Download dataset"):
            prog = st.progress(0.0)
            status = st.empty()

            def _progress(done: int, total: int | None):
                if total:
                    prog.progress(min(1.0, done / total))
                else:
                    prog.progress(0.0)

            def _log(msg: str):
                status.info(msg)

            try:
                path = ensure_dataset(url or None, sha256=(sha or None), progress=_progress, logger=_log)
                if path:
                    st.session_state["dataset_dir"] = path
                    st.success(f"Dataset ready at {path}")
                else:
                    st.error("Dataset not available. Configure a valid URL.")
            except Exception as e:
                st.error(f"Download failed: {e}")


def main():
    sidebar()
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
    # Provide a reliable link to the RL Training page that includes Q-Learning
    st.page_link("pages/3_RL_Training_Tabular.py", label="RL Training (Tabular + SB3)")


if __name__ == "__main__":
    main()

