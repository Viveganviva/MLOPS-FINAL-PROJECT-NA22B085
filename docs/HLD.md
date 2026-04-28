# High-Level Design

## Overview

This project is a market regime detection platform for daily equity data. It ingests Yahoo Finance history, builds regime-specific tabular features, trains three classifiers, serves live predictions through FastAPI, and wraps the whole system with Docker, DVC, MLflow, Prometheus, Grafana, and Airflow. The goal is not just prediction accuracy, but a reproducible operational loop: data in, model out, drift monitored, retraining triggered, and everything visible in dashboards.

## Architecture

```text
			 +-------------------+
			 |  Yahoo Finance    |
			 |  (yfinance)       |
			 +---------+---------+
					 |
					 v
			 +-------------------+
			 |  data/raw         |
			 |  data/simulation   |
			 +---------+---------+
					 |
					 v
			 +-------------------+
			 | Feature Engineering|
			 | src/feature_engineering.py
			 +---------+---------+
					 |
					 v
			 +-------------------+
			 | Training + MLflow  |
			 | src/train.py      |
			 | models/ mlruns/    |
			 +----+---------+----+
				 |         |
				 |         v
				 |   +------------+
				 |   | Grafana     |
				 |   | Prometheus  |
				 |   +------------+
				 v
		  +------------------------+
		  | FastAPI API            |
		  | api/main.py            |
		  +-----------+------------+
				    |
				    v
		   +------------------------+
		   | Frontend UI            |
		   | frontend/*.html        |
		   +-----------+------------+
					|
					v
		   +------------------------+
		   | Live Simulation        |
		   | src/simulate_live.py   |
		   +-----------+------------+
					|
					v
		   +------------------------+
		   | Drift + Retraining     |
		   | src/drift_monitor.py   |
		   | src/retraining_manager.py
		   +------------------------+
```

## Component Roles

- Frontend: Two static HTML dashboards. `index.html` is the operator prediction screen. `pipeline.html` is the operational control screen for health, drift, retraining, and history.
- FastAPI: The application layer. It exposes health, readiness, prediction, drift, retraining, pipeline status, prediction history, and service-health endpoints.
- MLflow: Tracks experiments, stores model artifacts, and provides the registry target for production model loading.
- Airflow: Holds the orchestration scaffold for data ingestion, feature engineering, and training runs.
- Prometheus: Scrapes API metrics on port 8001 and stores time-series counters, gauges, and histograms.
- Grafana: Reads Prometheus metrics and visualizes prediction rate, latency, confidence, drift, resource usage, and retraining activity.
- DVC: Reproduces the ingestion -> featurize -> train -> evaluate chain and tracks metrics and plot artifacts.

## Data Flow

1. `yfinance` downloads the training and simulation windows into `data/raw` and `data/simulation`.
2. `src/feature_engineering.py` reads the raw CSVs, creates regime-specific feature tables, and writes them into `data/processed`.
3. `src/train.py` reads the processed tables, trains the models, writes artifacts into `models/`, and logs runs to MLflow when available.
4. `api/main.py` loads the models, exposes prediction and drift endpoints, and writes prediction history to `data/simulation/prediction_log.csv`.
5. `frontend/index.html` and `frontend/pipeline.html` call the API and render the results for operators.
6. `src/drift_monitor.py` compares logged live features to baseline distributions and decides whether retraining is justified.
7. `src/retraining_manager.py` runs ingestion -> feature engineering -> train again when drift or manual intervention requires it.

## Design Choices

- XGBoost is used as the main model family because the regime problem is tabular and feature-engineered, not a true sequence-modeling problem at daily frequency. It is also easier to interpret through feature importance than an LSTM.
- Feature engineering is centralized in one module so training and inference use the same transformations. That reduces train/serve skew.
- DVC is used for repeatability. It makes the pipeline reconstructable from tracked dependencies and outputs.
- MLflow is used for model tracking and registry because the project needs a clear production artifact story, not just local pickle files.
- Prometheus and Grafana are used because the system needs operational visibility for latency, confidence, drift, and retraining behavior.
- FastAPI serves both JSON endpoints and static frontend files to keep the browser thin and avoid cross-origin problems in the UI.

## Technology Choices

| Tool | Purpose | Why Chosen | Alternative Considered |
|---|---|---|---|
| yfinance | Market data ingestion | Simple API, quick access to OHLCV history | Paid market data API |
| pandas / numpy | Feature engineering | Best fit for daily tabular finance data | Polars |
| XGBoost | Primary classifier | Strong on engineered tabular data, interpretable | LSTM / GRU |
| FastAPI | Service layer | Clean validation, simple JSON API, fast startup | Flask |
| MLflow | Experiment tracking | Registry + artifact tracking in one place | Custom file storage |
| DVC | Pipeline reproducibility | Tracks data, models, and metrics cleanly | Makefile only |
| Prometheus | Metrics scraping | Native counters, gauges, histograms | Cloud monitoring only |
| Grafana | Visualization | Flexible dashboarding for ops metrics | Static plots only |
| Airflow | Orchestration | Familiar DAG-based scheduling | Cron jobs |