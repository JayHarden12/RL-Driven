import os
import streamlit as st

from src.utils import get_dataset_dir


st.set_page_config(page_title="RL-Driven Building Energy Optimisation", layout="wide")


def sidebar():
    st.sidebar.title("Configuration")
    default_dir = get_dataset_dir() or os.path.abspath(".")
    dataset_dir = st.sidebar.text_input("Dataset folder", value=default_dir)
    if st.sidebar.button("Save dataset path"):
        st.session_state["dataset_dir"] = dataset_dir
        st.sidebar.success("Saved. Pages will use this path.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Set env var BDG2_DATASET_DIR to persist.")


def main():
    sidebar()
    st.title("Reinforcement-Learning–Driven Building Energy Optimisation")
    st.write(
        "Explore BDG2 data, engineer features, train a simple RL policy, and evaluate potential energy savings."
    )
    st.markdown(
        """
        - Use the sidebar to confirm the dataset path.
        - Navigate pages from the sidebar: Data Explorer, Feature Engineering, RL Training, Evaluation, and PRD Summary.
        - The RL environment is a didactic surrogate for setpoint control; refine it to suit your building.
        """
    )

    st.subheader("Quick Links")
    # Provide a reliable link to the RL Training page that includes Q-Learning
    st.page_link("pages/3_RL_Training_Tabular.py", label="RL Training (Tabular + SB3)")


if __name__ == "__main__":
    main()
