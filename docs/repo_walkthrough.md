# Repository Walkthrough

## Purpose

This file explains how the repository fits together from end to end, in the order a user or examiner would see it.

## What Runs First

1. `setup_project.py` is the bootstrap helper for the scaffold.
2. `src/data_ingestion.py` downloads and validates market data.
3. `src/feature_engineering.py` builds regime-specific features and baseline statistics.
4. `src/train.py` trains the models and writes artifacts into `models/`.
5. `dvc repro` records the pipeline outputs and metrics.
6. `docker-compose up --build -d` starts the API, frontend, MLflow, Prometheus, Grafana, and Airflow services.
7. `src/simulate_live.py` replays dates against the API.
8. The frontend screens and monitoring dashboards read the stored artifacts and metrics.

## File-by-File Guide

### Root files

- `README.md` is the short project overview.
- `params.yaml` is the central configuration file for windows, thresholds, split dates, and model settings.
- `requirements.txt` lists Python dependencies for local and Docker use.
- `conda.yaml` is the Conda environment spec if you want to recreate the environment another way.
- `MLproject` defines the project for MLflow-style execution.
- `dvc.yaml` declares the reproducible pipeline stages.
- `dvc.lock` is the recorded result of the pipeline state.
- `docker-compose.yml` runs the full service stack.
- `Dockerfile.api` builds the API container.
- `Dockerfile.frontend` builds the static UI container.

### `api/`

- `api/main.py` is the application entrypoint. It exposes the HTTP endpoints, records metrics, appends prediction logs, and serves the frontend files.
- `api/schemas.py` contains the Pydantic request and response models used by the API and tests.

### `src/`

- `src/data_ingestion.py` downloads OHLCV data, validates it, writes raw CSVs, and creates baseline statistics.
- `src/feature_engineering.py` computes indicators, labels regimes, and saves processed feature tables.
- `src/train.py` splits the feature tables by date, scales them, trains models, evaluates them, and logs artifacts.
- `src/predict.py` loads the trained artifacts, downloads recent data, rebuilds features, and returns predictions.
- `src/drift_monitor.py` compares recent feature snapshots against baselines and decides whether retraining is justified.
- `src/retraining_manager.py` sequences ingestion, feature engineering, and training when retraining is triggered.
- `src/monitoring.py` defines Prometheus metrics and the background system collector.
- `src/simulate_live.py` replays trading dates against the API and triggers drift checks on a schedule.
- `src/save_dvc_metrics.py` rebuilds evaluation metrics and confusion-matrix CSVs for DVC.

### `frontend/`

- `frontend/index.html` is the main prediction screen.
- `frontend/pipeline.html` is the operational pipeline screen.

### `airflow_dags/`

- `airflow_dags/regime_pipeline_dag.py` is the orchestration scaffold for running ingestion, featurization, and training as DAG tasks.

### `grafana/`

- `grafana/provisioning/datasources/prometheus.yaml` auto-connects Grafana to Prometheus.
- `grafana/provisioning/dashboards/dashboard.yaml` tells Grafana where to load dashboards from.
- `grafana/provisioning/dashboards/regime_dashboard.json` defines the operational dashboard.

### `prometheus/`

- `prometheus/prometheus.yml` tells Prometheus which targets to scrape.

### `tests/`

- `tests/test_ingestion.py` checks ingestion and baseline creation.
- `tests/test_features.py` checks RSI and feature output consistency.
- `tests/test_predict.py` checks the prediction contract.
- `tests/test_api.py` checks the main API endpoints.

### `docs/`

- `docs/HLD.md` explains the system architecture.
- `docs/LLD.md` explains API contracts, module dependencies, and error codes.
- `docs/user_manual.md` explains the app for non-technical users.
- `docs/architecture_diagram.md` captures the container layout and persistence model.
- `docs/repo_walkthrough.md` is this file.

### `Experiments/`

- The notebooks and notes in `Experiments/` are the exploratory work that informed the final pipeline.

## Execution Flow

1. Ingestion writes raw data to `data/raw/` and simulation data to `data/simulation/`.
2. Feature engineering reads `data/raw/` and writes `data/processed/` plus the baseline file in `data/baselines/`.
3. Training reads `data/processed/` and writes model artifacts into `models/`.
4. The API loads `models/` and exposes `/health`, `/ready`, `/predict`, `/drift/{regime}`, `/retrain`, `/ground-truth`, `/pipeline-status`, `/prediction-history`, and `/service-health/{service_name}`.
5. The frontend calls those endpoints and displays the results.
6. Prometheus scrapes the API metrics endpoint on 8001.
7. Grafana visualizes those metrics.
8. DVC records the reproducible pipeline outputs.

## What Gets Stored

- Raw downloads: `data/raw/`
- Processed features: `data/processed/`
- Baselines and training report: `data/baselines/`
- Simulation prediction log: `data/simulation/prediction_log.csv`
- Trained models and schemas: `models/`
- MLflow runs: `mlruns/`
- Logs: `logs/`
- Grafana state: `grafana_data` volume

## How the Pieces Connect

The important design rule is that the same feature-engineering functions are used in both training and inference. That keeps the model input shape stable and avoids train/serve skew. The API is the hub for runtime interactions, while DVC and MLflow cover reproducibility and experiment tracking. Prometheus and Grafana cover observability, and Airflow exists as the orchestration layer for repeatable jobs.

## Viva-Level Explanation

If you are asked to explain the project, the safest answer is:

"This repository turns daily market data into a reproducible MLOps system. It downloads data, builds regime features, trains models, serves predictions, monitors drift, and supports retraining. The Docker stack makes it deployable, DVC makes it reproducible, MLflow tracks experiments, and the UI plus dashboards make it usable."
