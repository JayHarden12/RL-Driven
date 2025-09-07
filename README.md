RL-Driven Building Energy Optimisation (Streamlit App)

Overview

- Streamlit app to explore the Building Data Genome 2 (BDG2) dataset, engineer features, train RL agents, and evaluate potential energy savings.
- Includes a lightweight surrogate environment for HVAC setpoint control; runs entirely from historical data without a physics simulator.

Quick Start

1) Python 3.9–3.11 recommended.
2) Create a virtual environment and install requirements:
   - Windows (PowerShell)
     - `python -m venv .venv`
     - `.venv\\Scripts\\Activate.ps1`
     - `pip install -r requirements.txt`
3) Launch the app:
   - `streamlit run app.py`

Dataset

- Place BDG2 CSVs under `Building Data Genome Project 2 dataset/` at the repo root (present in this workspace).
- Expected files: `electricity.csv`, `weather.csv`, `metadata.csv` (others like `gas.csv`, `chilledwater.csv` optional).

Pages

- Overview: App description and configuration.
- Data Explorer: Load, filter, and visualise time series by building and period.
- Feature Engineering: Aggregate and add weather/time features for modelling.
- RL Training: Train a policy using Stable-Baselines3 (PPO/DQN) or Tabular Q-Learning.
- Evaluation: Compare baseline vs RL policy on holdout, view KPIs (energy, cost, emissions) and charts.
- PRD Summary: Best-effort text extraction/preview of the attached PRD `.docx`.

Q-Learning (New)

- A tabular Q-Learning agent is included under `src/agents/q_learning.py` and can be selected on the RL Training page as "Q-Learning (tabular)".
- The agent discretizes the 6D observation into bins and learns a sparse Q-table with epsilon-greedy exploration.
- Configurable hyperparameters exposed in the UI:
  - Bins per feature, Alpha (learning rate), Gamma (discount), Epsilon start/end, Epsilon decay steps.
- Trained models are saved to `artifacts/<building>_Q-Learning.pkl` and can be evaluated on the Evaluation page like PPO/DQN.

Notes

- The surrogate environment is intentionally simple for clarity and speed. It adjusts a cooling setpoint delta in a small discrete range and receives a reward balancing energy reduction and comfort.
- You can refine the surrogate model (e.g., fit a consumption regressor per building with weather + time features) from the Feature Engineering page.
- The app tries to auto-locate the dataset folder; override it in the sidebar if needed.

