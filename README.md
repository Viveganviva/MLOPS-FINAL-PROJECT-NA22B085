# Market Regime Detection — MLOps Final Project

> **NA22B085** · End-to-end MLOps pipeline for next-day market regime classification using daily equity data, served through a fully containerised stack with experiment tracking, monitoring, drift detection, and automated retraining.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Environment Variables](#environment-variables)
6. [Running the Pipeline Manually](#running-the-pipeline-manually)
7. [Starting Docker](#starting-docker)
8. [Service URLs and What to Check](#service-urls-and-what-to-check)
9. [Running Tests](#running-tests)
10. [Running the Live Simulation](#running-the-live-simulation)
11. [Stopping the Stack](#stopping-the-stack)
12. [Repository Structure](#repository-structure)
13. [Artifacts](#artifacts)
14. [Model Overview](#model-overview)
15. [MLOps Tools Summary](#mlops-tools-summary)

---

## What This Project Does

This system classifies the current market state across three orthogonal regime dimensions using daily OHLCV data freely available from Yahoo Finance:

| Regime | Labels | Description |
|---|---|---|
| **Trend vs Mean-Revert** | `Trending` / `MeanReverting` | Is the market showing persistent directional movement or oscillating around a fair value? |
| **Volatility** | `HighVol` / `LowVol` | Is daily price variance elevated (risk-on caution) or suppressed (calm conditions)? |
| **Market Direction** | `Bull` / `Bear` | Is the broad market in a structurally positive or negative phase? |

A user selects a ticker and date in the browser UI, clicks **Analyze**, and receives all three predictions with confidence scores in under 200 ms. The full MLOps loop — ingestion → feature engineering → training → serving → drift detection → retraining — is wired across DVC, MLflow, Airflow, Prometheus, and Grafana.

---

## System Architecture

```
                        ┌──────────────────────────────────────┐
                        │           Browser  (port 80)          │
                        │  frontend/index.html                  │
                        │  frontend/pipeline.html               │
                        └─────────────────┬────────────────────┘
                                          │  HTTP  fetch()
                                          ▼
                        ┌──────────────────────────────────────┐
                        │     Nginx  (Dockerfile.frontend)      │
                        │   /pipeline  →  pipeline.html         │
                        │   /*         →  index.html (fallback) │
                        └─────────────────┬────────────────────┘
                                          │  REST
                                          ▼
          ┌───────────────────────────────────────────────────────┐
          │              FastAPI  api/main.py  (port 8000)        │
          │   /health  /ready  /predict  /drift  /retrain         │
          │   /ground-truth  /pipeline-status  /prediction-history│
          └──┬─────────────────────────────────────────┬──────────┘
             │                                         │
             ▼                                         ▼
  ┌─────────────────────┐                  ┌─────────────────────┐
  │  src/predict.py      │                  │  Prometheus metrics  │
  │  src/drift_monitor.py│                  │  port 8001 → 9090    │
  │  src/retraining_     │                  └──────────┬──────────┘
  │    manager.py        │                             │
  └──────────┬───────────┘                             ▼
             │                              ┌─────────────────────┐
             ▼                              │  Grafana  (port 3000)│
  ┌─────────────────────┐                  │  regime_dashboard    │
  │  src/feature_        │                  └─────────────────────┘
  │    engineering.py    │
  └──────────┬───────────┘
             │
             ▼
  ┌─────────────────────┐        ┌─────────────────────┐
  │  data/raw/           │        │  models/  mlruns/    │
  │  data/processed/     │◄──────►│  src/train.py        │
  │  data/baselines/     │        │  MLflow  (port 5000)  │
  │  data/simulation/    │        └─────────────────────┘
  └─────────────────────┘
             ▲
             │  BashOperator curl calls
  ┌─────────────────────┐
  │  Airflow  (port 8080)│
  │  regime_pipeline DAG │
  └─────────────────────┘
```

**Data flow:**  
yfinance → `data/raw/` → feature engineering → `data/processed/` → training → `models/` + MLflow → FastAPI loads models → prediction on request → Prometheus scrapes metrics → Grafana renders dashboard → drift monitor reads `prediction_log.csv` → retraining manager closes the loop.

---

## Prerequisites

Install these before anything else:

| Tool | Version | Install link |
|---|---|---|
| **Docker Desktop** | Latest | https://www.docker.com/products/docker-desktop |
| **Python** | 3.10 – 3.12 | https://www.python.org/downloads/ |
| **Git** | Any | https://git-scm.com/ |
| **DVC** | `pip install dvc` | https://dvc.org/doc/install |

Everything else (MLflow, XGBoost, FastAPI, yfinance, etc.) is installed from `requirements.txt`.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Viveganviva/MLOPS-FINAL-PROJECT-NA22B085.git
cd MLOPS-FINAL-PROJECT-NA22B085

# 2. Create and activate a Python environment
conda create -n regime_env python=3.10 -y
conda activate regime_env
# or: python -m venv .venv && .venv\Scripts\activate (Windows)

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create your .env file (see Environment Variables section)
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux
# then edit .env with your SMTP credentials

# 5. Run the data pipeline locally (before starting Docker)
python src/data_ingestion.py
python src/feature_engineering.py
python src/train.py --force

# 6. Start the full Docker stack
docker-compose up --build -d

# 7. Open the UI
# http://localhost
```

---

## Environment Variables

Create a file called `.env` in the project root. **Never commit this file — it is already in `.gitignore`.**

```env
# Email alerting (set to false to disable)
ALERT_EMAIL_ENABLED=false
ALERT_EMAIL_TO=your@email.com
SMTP_USER=yourgmail@gmail.com
SMTP_PASSWORD=your_16_char_app_password
```

**Getting a Gmail App Password:**  
Google Account → Security → 2-Step Verification → App passwords → Generate one for "Mail". Use the 16-character code, not your regular Gmail password.

Docker Compose reads `.env` automatically and injects these into the API container as environment variables. The SMTP credentials are never hardcoded in any source file.

---

## Running the Pipeline Manually

Run these in order the first time, and whenever you need to retrain:

### Step 1 — Download market data

```bash
python src/data_ingestion.py
```

Downloads SPY, QQQ, IWM, GLD, TLT, VIX from Yahoo Finance (2010–2022) into `data/raw/`.  
Also downloads the held-out simulation window (2023–2024) into `data/simulation/`.  
Prints a validation table showing PASS/FAIL per ticker.

### Step 2 — Build features

```bash
python src/feature_engineering.py
```

Reads `data/raw/`, computes all regime-specific indicators and labels, writes three CSVs to `data/processed/`, and saves baseline statistics to `data/baselines/feature_baselines.json`.

### Step 3 — Train models and log to MLflow

First, start MLflow so runs are visible in the UI:

```bash
# In a separate terminal — keep it running
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns/artifacts
```

Then in your main terminal:

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000   # Windows CMD
# export MLFLOW_TRACKING_URI=http://localhost:5000  # Mac/Linux

python src/train.py --force
```

Training takes 3–6 minutes. At the end it prints a results table with F1 scores per regime and MLflow run IDs.  
Add `--force` to bypass the 7-day freshness check and always retrain.

### Step 4 — Record pipeline state with DVC

```bash
dvc repro          # re-runs only changed stages
dvc metrics show   # prints F1 scores
dvc dag            # prints the pipeline dependency graph
```

---

## Starting Docker

```bash
# Build and start all containers in the background
docker-compose up --build -d

# Check all containers are running
docker-compose ps

# Watch live logs from the API container
docker-compose logs -f api

# Rebuild only the API after code changes (fast)
docker-compose up --build api -d
```

Expected output of `docker-compose ps` — all containers should show **Up** or **Up (healthy)**:

```
NAME                STATUS
regime_api          Up (healthy)
regime_frontend     Up
regime_mlflow       Up (healthy)
regime_prometheus   Up
regime_grafana      Up
regime_airflow      Up
```

---

## Service URLs and What to Check

| Service | URL | Login | What to check |
|---|---|---|---|
| **Frontend UI** | http://localhost | — | Main prediction screen and pipeline screen |
| **API Swagger** | http://localhost:8000/docs | — | All endpoint definitions; try `/health` and `/predict` |
| **API Health** | http://localhost:8000/health | — | Should return `{"status":"ok","models_loaded":{"trend":true,"vol":true,"bull_bear":true}}` |
| **MLflow** | http://localhost:5000 | — | Experiments tab → 3 experiments with runs; Models tab → 3 registered models |
| **Airflow** | http://localhost:8080 | `admin` / `admin` | DAGs list → `regime_pipeline` → Graph tab shows the three-task flow |
| **Prometheus** | http://localhost:9090 | — | Status → Targets → `regime_detection_api` should be **UP** |
| **Grafana** | http://localhost:3000 | `admin` / `admin` | Dashboards → regime-dashboard → 8 operational panels |

### Verifying predictions via curl

```bash
# Windows CMD
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"ticker\":\"SPY\",\"regime_types\":[\"trend\",\"vol\",\"bull_bear\"]}"

# Mac/Linux
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"SPY","regime_types":["trend","vol","bull_bear"]}'
```

Expected: JSON with `predicted_label`, `confidence`, and `inference_latency_ms` for each regime.

### Triggering drift checks

```bash
curl http://localhost:8000/drift/trend
curl http://localhost:8000/drift/vol
curl http://localhost:8000/drift/bull_bear
```

### Manually triggering retraining

```bash
curl -X POST "http://localhost:8000/retrain?triggered_by=manual"
```

### Grafana setup (if dashboard doesn't auto-load)

1. Open http://localhost:3000 → login `admin` / `admin`  
2. Left sidebar → Connections → Data Sources → verify **Prometheus** with URL `http://prometheus:9090` shows green  
3. If dashboard is missing: Dashboards → New → Import → upload `grafana/provisioning/dashboards/regime_dashboard.json`

### Airflow setup

1. Open http://localhost:8080 → login `admin` / `admin`  
2. Find `regime_pipeline` in the DAG list  
3. Toggle the pause switch to **ON**  
4. Click the ▶ **Trigger DAG** button to run it manually  
5. Click the DAG name → **Graph** tab to see the three-task dependency view

---

## Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v --tb=short

# Save the test report
pytest tests/ -v --tb=short 2>&1 | tee docs/test_report.txt

# Run a specific test file
pytest tests/test_api.py -v
pytest tests/test_features.py -v
```

Test results are saved to `docs/test_report.txt`. The test suite covers:
- Data ingestion and validation
- Feature engineering correctness (RSI bounds, NaN cleanup, leakage guard)
- Prediction contract (required keys, confidence range, label validity)
- API endpoints (health, readiness, predict, invalid ticker handling)

---

## Running the Live Simulation

The simulation replays the held-out 2023–2024 SPY data against the live API, one trading day at a time. The API must be running (Docker up) before starting the simulation.

### Basic usage

```bash
python src/simulate_live.py
```

### Full options

```bash
python src/simulate_live.py \
  --ticker SPY \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --speed 2.0 \
  --regimes trend,vol,bull_bear \
  --drift-every-n-days 20 \
  --api-url http://127.0.0.1:8000 \
  --timeout 30
```

| Parameter | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | Ticker symbol to replay |
| `--start-date` | `2023-01-01` | First simulation date |
| `--end-date` | `2024-12-31` | Last simulation date |
| `--speed` | `0.5` | Seconds to wait between each trading day |
| `--regimes` | `trend,vol,bull_bear` | Comma-separated list of regimes to predict |
| `--drift-every-n-days` | `5` | How often to run a drift check (every N days) |
| `--api-url` | `http://127.0.0.1:8000` | API base URL |
| `--timeout` | `30` | HTTP request timeout in seconds |

### What to expect

Each line shows one trading day:

```
[2023-06-15] SPY | trend=MeanReverting (0.97, 1850ms) | vol=HighVol (0.93, 110ms) | bull_bear=Bull (0.99, 95ms)
```

Every N days a drift check runs. If drift exceeds the threshold, retraining is triggered automatically via the API.

Predictions are appended to `data/simulation/prediction_log.csv` in real time.  
Stop the simulation at any time with **Ctrl+C**. A summary is printed on exit.

While the simulation is running, watch http://localhost:3000 (Grafana) — the prediction counter, latency histogram, and confidence gauges update live.

---

## Stopping the Stack

```bash
# Stop all containers (data and volumes are preserved)
docker-compose down

# Stop and delete all persistent volumes (wipes Grafana state)
docker-compose down -v

# Stop a specific container
docker-compose stop api

# Restart a specific container
docker-compose restart airflow
```

---

## Repository Structure

```
MLOPS-FINAL-PROJECT-NA22B085/
│
├── api/                          # FastAPI application
│   ├── main.py                   # All endpoints: health, predict, drift, retrain, history
│   └── schemas.py                # Pydantic request/response models
│
├── src/                          # Core pipeline modules
│   ├── data_ingestion.py         # Download, validate, and save raw market CSVs
│   ├── feature_engineering.py    # Shared feature builder (used by training AND inference)
│   ├── train.py                  # Time-split, train XGBoost, log to MLflow, register model
│   ├── predict.py                # Live inference: fetch data → features → scale → predict
│   ├── drift_monitor.py          # KL-divergence drift detection against baselines
│   ├── retraining_manager.py     # Orchestrates ingestion → features → train on retrain trigger
│   ├── monitoring.py             # Prometheus metric definitions + system metrics collector
│   ├── simulate_live.py          # Replay 2023-2024 data against the API day by day
│   └── save_dvc_metrics.py       # Writes dvc_metrics.json and confusion matrix CSVs for DVC
│
├── airflow_dags/
│   └── regime_pipeline_dag.py    # Airflow DAG: health_check → drift_check → trigger_retrain
│
├── frontend/
│   ├── index.html                # Main prediction screen (dark financial terminal UI)
│   ├── pipeline.html             # Pipeline management and monitoring screen
│   └── nginx.conf                # Custom nginx routing (/pipeline → pipeline.html)
│
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yaml   # Auto-connects Grafana to Prometheus on startup
│       └── dashboards/
│           ├── dashboard.yaml    # Tells Grafana where to load dashboards from
│           └── regime_dashboard.json  # 8-panel operational dashboard (auto-provisioned)
│
├── prometheus/
│   └── prometheus.yml            # Scrape config: pulls metrics from api:8001 every 15s
│
├── tests/
│   ├── test_ingestion.py         # ING-01 through ING-03
│   ├── test_features.py          # FEAT-01 through FEAT-04
│   ├── test_predict.py           # PRED-01 through PRED-03
│   └── test_api.py               # API-01 through API-04
│
├── Experiments/
│   ├── trend_regime_eda.ipynb    # Notebook: Hurst, VR, autocorr labelling + XGBoost
│   ├── volatility_regime_detection.ipynb  # Notebook: GARCH, VIX, realized vol labelling
│   └── bull_vs_bear.ipynb        # Notebook: drawdown rule, cross-asset features, LSTM
│
├── data/                         # Generated — not committed to git (except baselines)
│   ├── raw/                      # Downloaded OHLCV CSVs (SPY, QQQ, IWM, GLD, TLT, VIX)
│   ├── processed/                # Feature-engineered datasets per regime
│   ├── baselines/                # feature_baselines.json and training_report.json
│   └── simulation/               # SPY_sim.csv, VIX_sim.csv, prediction_log.csv
│
├── models/                       # Generated — not committed to git
│   ├── trend_model.pkl           # Trained XGBoost classifier (trend regime)
│   ├── trend_scaler.pkl          # StandardScaler fitted on trend training data
│   ├── trend_feature_columns.json  # Ordered list of training feature names
│   ├── trend_classes.json        # Class label mapping [0, 1] → ['MeanReverting', 'Trending']
│   ├── trend_confusion_matrix.png
│   ├── trend_feature_importance.png
│   └── ... (same pattern for vol_ and bull_bear_)
│
├── mlruns/                       # MLflow tracking store — experiments, runs, registry
│   ├── mlflow.db                 # SQLite backend for the MLflow server
│   ├── models/                   # Registered model entries (TrendRegimeModel etc.)
│   └── <experiment_id>/          # One folder per experiment with metrics, params, tags
│
├── artifacts/                    # Screenshots and PDF for project report
│   ├── eda1.png – eda4.png       # EDA notebook screenshots
│   ├── ui_main.png               # Main UI screenshot
│   ├── ui_pipeline.png           # Pipeline screen screenshot
│   ├── ui_mlflow.png             # MLflow experiments screenshot
│   ├── ui_airflow.png            # Airflow DAG graph screenshot
│   ├── ui_grafana.png            # Grafana dashboard screenshot
│   ├── ui_prometheus.png         # Prometheus targets screenshot
│   └── MLOPS_FINAL_PROJECT_REPORT_NA22B085.pdf
│
├── docs/
│   ├── HLD.md                    # High-level design: architecture and tool rationale
│   ├── LLD.md                    # Low-level design: endpoint spec and IO schemas
│   ├── architecture_diagram.md   # Block-level architecture explanation for viva
│   ├── repo_walkthrough.md       # File-by-file guide to the repository
│   ├── test_plan.md              # Test objectives, cases, and acceptance criteria
│   ├── test_report.txt           # Output of pytest run
│   └── user_manual.md            # Non-technical guide to using the application
│
├── logs/                         # Runtime logs (gitignored except structure)
│   ├── ingestion.log
│   ├── features.log
│   ├── training.log
│   └── dag_id=regime_pipeline/   # Airflow task logs
│
├── dvc_plots/                    # Confusion matrix CSVs for dvc plots show
│   ├── confusion_matrix_trend.csv
│   ├── confusion_matrix_vol.csv
│   └── confusion_matrix_bull_bear.csv
│
├── docker-compose.yml            # Full service topology (api, frontend, mlflow, prometheus, grafana, airflow)
├── Dockerfile.api                # API container: python:3.10-slim + requirements
├── Dockerfile.frontend           # Frontend container: nginx:alpine + custom routing
├── MLproject                     # MLflow Projects entry points (ingest, featurize, train, simulate)
├── conda.yaml                    # Conda environment spec for MLflow Projects
├── dvc.yaml                      # DVC pipeline: ingest → featurize → train → evaluate
├── dvc.lock                      # DVC artifact hashes for reproducibility
├── dvc_metrics.json              # F1 scores per regime (read by dvc metrics show)
├── params.yaml                   # Centralized config: windows, splits, hyperparameters, thresholds
├── requirements.txt              # Python dependencies
├── .env                          # Local secrets — NEVER commit (gitignored)
└── .gitignore
```

---

## Artifacts

The `artifacts/` folder contains all screenshots used in the project report. After making any change to the UI or tooling, replace the relevant screenshots:

| File | Content |
|---|---|
| `eda1.png` | Regime label overlay on SPY price chart (from trend notebook) |
| `eda2.png` | Feature correlation heatmap |
| `eda3.png` | XGBoost feature importance chart |
| `eda4.png` | Equity curves: strategy vs buy-and-hold backtest |
| `ui_main.png` | Main prediction screen with regime cards filled |
| `ui_pipeline.png` | Pipeline management screen |
| `ui_mlflow.png` | MLflow experiments with three regime runs |
| `ui_airflow.png` | Airflow DAG graph view |
| `ui_grafana.png` | Grafana dashboard with live panels |
| `ui_prometheus.png` | Prometheus targets showing API as UP |
| `MLOPS_FINAL_PROJECT_REPORT_NA22B085.pdf` | Full project report |

---

## Model Overview

Three independent XGBoost classifiers are trained, one per regime dimension.

| Regime | Features | Labelling method | Test F1 | Test Accuracy |
|---|---|---|---|---|
| Trend / MeanRev | RSI, MACD, Bollinger width, ATR, rolling stats, autocorr | Majority vote: Hurst exponent + variance ratio + lag-1 autocorrelation | **0.59** | 0.71 |
| Volatility | Realized vol, Parkinson vol, Garman-Klass vol, VIX level, GARCH conditional vol | Majority vote: rolling vol threshold + VIX threshold + GARCH threshold | **0.86** | 0.90 |
| Bull / Bear | SPY technicals + cross-asset (TLT, GLD, QQQ, IWM, VIX) | Peak-to-trough 20% drawdown rule (standard institutional definition) | **0.69** | 0.78 |

Training window: 2010-01-01 → 2020-12-31  
Validation window: 2021-01-01 → 2021-12-31  
Test window: 2022-01-01 → 2022-12-31  
Simulation (held-out, never seen by model): 2023-01-01 → 2024-12-31

**Why these F1 scores are correct:** Financial return series are close to a random walk. An F1 of 0.59 for trend prediction represents a measurable statistical edge over a 0.50 random baseline on a genuinely held-out period that included the 2022 Fed rate-hike regime transitions — a market environment the model had never seen.

---

## MLOps Tools Summary

| Tool | Role in this project |
|---|---|
| **DVC** | Tracks raw data, processed features, and trained models; `dvc repro` rebuilds the pipeline from any commit; `dvc metrics diff` compares F1 across commits |
| **MLflow** | Logs every training run with hyperparameters, metrics, confusion matrix, and feature importance artifacts; hosts the model registry so the API always loads the `production` alias |
| **Airflow** | Schedules the pipeline (daily at 6 PM weekdays); DAG calls API endpoints for drift check and retraining so it is version-agnostic |
| **Prometheus** | Scrapes 10 custom metrics from port 8001 every 15 s: prediction counters, latency histograms, confidence gauges, drift scores, CPU/memory, error rate |
| **Grafana** | Renders 8-panel operational dashboard auto-provisioned from JSON; shows near-real-time updates during simulation |
| **Docker Compose** | Runs 6 containers with environment parity across dev and demo; volumes preserve all state across restarts |
| **FastAPI** | REST API with Pydantic validation, auto-generated Swagger docs at `/docs`, and middleware for Prometheus request tracking |

---

## Common Commands Reference

```bash
# Full pipeline from scratch
python src/data_ingestion.py
python src/feature_engineering.py
python src/train.py --force

# DVC
dvc repro
dvc dag
dvc metrics show
dvc metrics diff HEAD~1

# Docker
docker-compose up --build -d          # start everything
docker-compose ps                     # check status
docker-compose logs -f api            # stream API logs
docker-compose logs -f airflow        # stream Airflow logs
docker-compose up --build api -d      # rebuild just the API
docker-compose restart airflow        # restart one service
docker-compose down                   # stop, keep volumes
docker-compose down -v                # stop, delete volumes

# Simulation
python src/simulate_live.py --speed 1 --ticker SPY --drift-every-n-days 10

# Tests
pytest tests/ -v --tb=short

# Manual API calls
curl http://localhost:8000/health
curl http://localhost:8000/drift/vol
curl -X POST "http://localhost:8000/retrain?triggered_by=manual"
```
```
