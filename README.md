# Market Regime Detection

Professional MLOps scaffold for a market regime detection system covering data ingestion, feature engineering, model training, live simulation, API serving, monitoring, and dashboarding.

## Project Layout

- `src/` contains the core pipeline scripts.
- `api/` exposes the prediction service.
- `frontend/` hosts the operator UI.
- `airflow_dags/` holds orchestration definitions.
- `docs/` contains architecture and delivery documentation.

## Bootstrap

Run `python setup_project.py` once to initialize the scaffold state, including the prediction log header and package marker files.

Then continue with data ingestion and the rest of the workflow.