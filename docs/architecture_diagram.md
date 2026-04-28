# Architecture Diagram and Viva Journal

## Purpose

This document explains the current shape of the project after the recent pipeline and frontend updates. It is written as a study note for a viva, so it focuses on how the system works, why each part exists, and how data moves from raw market downloads to the final browser UI.

The key design idea is simple: all market-data handling is centralized in the Python pipeline, while the browser only consumes the API and static HTML assets. That separation keeps model code reusable, makes the frontend lightweight, and avoids mixing presentation concerns with data-processing logic.

## High-Level View

```text
                          +--------------------------------------+
                          |            Browser UI                |
                          | frontend/index.html                  |
                          | frontend/pipeline.html               |
                          +------------------+-------------------+
                                             |
                                             v
                          +--------------------------------------+
                          |            Nginx Frontend            |
                          | Dockerfile.frontend                  |
                          | frontend/nginx.conf or inline config |
                          +------------------+-------------------+
                                             |
                                             v
                          +--------------------------------------+
                          |               FastAPI                |
                          | api/main.py                          |
                          | /health /predict /pipeline ...       |
                          +-----------+--------------------------+
                                      |                          |
                                      |                          v
                                      |             +------------------------+
                                      |             | Prometheus Metrics     |
                                      |             | port 8001 -> 9090      |
                                      |             +-----------+------------+
                                      |                         |
                                      v                         v
                        +-----------------------------+   +----------------------+
                        | Prediction History / Logs   |   | Grafana Dashboard    |
                        | data/simulation             |   | regime_dashboard.json|
                        +--------------+--------------+   +----------------------+
                                       |
                                       v
                        +-----------------------------+
                        | Prediction Engine            |
                        | src/predict.py               |
                        +--------------+--------------+
                                       |
                                       v
                        +-----------------------------+
                        | Shared Feature Logic         |
                        | src/feature_engineering.py   |
                        +--------------+--------------+
                                       |
                                       v
                        +-----------------------------+
                        | Raw and Processed Market Data|
                        | data/raw                     |
                        | data/processed               |
                        | data/baselines               |
                        +--------------+--------------+
                                       |
                                       v
                        +-----------------------------+
                        | Training and Registry       |
                        | src/train.py                |
                        | models/                     |
                        | mlruns/                     |
                        +--------------+--------------+
                                       |
                                       v
                        +-----------------------------+
                        | Orchestration Scaffold      |
                        | airflow_dags/               |
                        +-----------------------------+
```

## What Each Layer Does

### 1. Browser UI

The browser only renders the operator-facing interface. It does not compute signals itself. Its responsibilities are to collect input, call the API, and display results. That keeps the user experience responsive and prevents the browser from depending on training logic.

The frontend now uses hostname-aware API routing. In practice, that means the same HTML can run on localhost during development and still point to the correct API host when opened from another machine on the network. This is important for demos and containerized deployments because hardcoding `localhost` breaks remote access patterns.

### 2. Nginx Frontend Layer

The frontend container is a thin static web server. Nginx serves the HTML files and handles routing rules so browser navigation behaves correctly.

The custom routing does two things:

1. It serves `/pipeline` by returning `pipeline.html` directly.
2. It falls back to `index.html` for any other missing route so the single-page UI still loads cleanly.

This matters because frontend routes can be reached by refreshing or deep-linking. Without the fallback, Nginx would return a 404 even though the SPA itself knows how to render the page.

### 3. FastAPI API Layer

The API is the central application boundary. It handles health checks, prediction requests, and any supporting service endpoints the frontend needs.

The API does not train models. Instead, it loads saved artifacts from disk or registry, runs inference, and returns structured JSON. That separation is deliberate: training is slow and batch-oriented, while inference must stay fast and predictable.

### 4. Prediction Engine

`src/predict.py` is the live inference path. It downloads recent market data, prepares features using the shared feature-engineering module, scales the feature row with the saved scaler, and asks the trained model for the final prediction.

The recent yfinance normalization work is especially important here. Newer yfinance versions can return MultiIndex columns even for single-ticker downloads. If those columns are not flattened, downstream code may fail when it expects names like `Close`, `Open`, and `Volume`. The inference path now normalizes that shape before feature generation, which makes live prediction safer and less version-sensitive.

### 5. Shared Feature Logic

`src/feature_engineering.py` is the core feature library. It is shared by training and inference so both sides compute the same inputs from the same rules.

This module has three major jobs:

1. It labels regimes.
2. It builds features for each regime family.
3. It protects the pipeline from shape and schema surprises.

The new `_normalise_ohlcv()` helper is part of that last job. It strips away yfinance MultiIndex columns and removes duplicates after flattening. That means every downstream function can assume a stable OHLCV schema, which is critical when the same code is used both offline and live.

### 6. Data Storage

The data folders serve different roles:

- `data/raw/` stores downloaded market CSVs.
- `data/processed/` stores engineered or cleaned outputs.
- `data/baselines/` stores reference artifacts for monitoring or comparison.
- `data/simulation/` stores prediction history and simulated outputs.
- `models/` stores trained models and their associated metadata.
- `mlruns/` stores MLflow tracking artifacts.
- `logs/` stores pipeline logs.

These folders matter because they are the boundary between code and persisted state. The pipeline can be rebuilt from them, and the model-serving side can recover after restarts.

### 7. Training and Registry

`src/train.py` is the training entry point. It takes prepared data, builds regime-specific models, evaluates them, and registers or saves the resulting artifacts.

Training is separate from inference for a reason: the training workflow can afford to be slower, more verbose, and more experimental. The output of training is not a UI response, but a set of artifacts that inference will reuse later.

### 8. Orchestration Scaffold

The Airflow folder exists for scheduled or repeatable pipeline execution. Even when the DAGs are minimal, the structure signals that the project is designed to support automated runs, not just manual notebooks.

## Updated Data Flow

The operational flow now looks like this:

1. Market data is downloaded from yfinance.
2. The raw frame is normalized so the columns are always usable by the rest of the pipeline.
3. Feature engineering builds regime-specific signals.
4. Training consumes those signals and writes model artifacts.
5. Prediction loads the artifacts, downloads recent data, normalizes it again, and produces live output.
6. The API returns structured results to the browser.
7. The frontend displays the result and routes the user through the static pages served by Nginx.

The most important point is that normalization happens at the boundaries. That means the system is resilient even if yfinance changes how it returns columns in future releases.

## Regime Pipeline Walkthrough

### Trend Regime

The trend pipeline classifies whether the market is trending or mean-reverting. It uses label-safe signals such as Hurst exponent, variance ratio, and autocorrelation. Those signals are designed for regime detection rather than short-term prediction leakage.

The feature builder then adds momentum, rolling statistics, MACD, Bollinger-derived features, ATR, and related measures. The result is a row-aligned dataset with a final regime label attached.

### Volatility Regime

The volatility pipeline tries to detect high-volatility versus low-volatility conditions. It combines realized volatility, VIX-based stress, and conditional volatility estimates.

This is the kind of regime where a small data-shape bug can break the flow easily, because one missing `Close` column or one duplicated MultiIndex level can stop the whole calculation chain. That is why the normalization helper is valuable here.

### Bull/Bear Regime

The bull/bear pipeline uses SPY plus cross-asset confirmation from QQQ, IWM, GLD, TLT, and VIX. The idea is that market regime is not just about one ETF; breadth and safety-rotation assets often provide the confirming context.

The feature set includes relative strength, momentum, rolling volatility, bond and gold context, and VIX behavior. That makes the model more robust than a single-asset price-only classifier.

## Why the yfinance Normalization Matters

yfinance changed behavior in a way that can surprise downstream code. A call like `download('SPY', ...)` can return columns that look like a two-level index instead of a simple flat table. If a function expects `df['Close']`, it may fail or silently behave incorrectly.

The project now treats this as a boundary concern:

- In inference, the download result is normalized immediately after fetching.
- In feature engineering, every public entry point normalizes again so the module is safe even if called directly.
- Duplicate columns are dropped after flattening so the frame stays deterministic.

That pattern is good engineering practice because it moves the compatibility fix to the place where the external dependency enters the codebase.

## Docker and Frontend Routing Notes

The frontend container is intentionally simple. It copies the HTML assets into Nginx’s web root and then installs a route configuration.

The routing rules are important for two reasons:

1. `/pipeline` should resolve to `pipeline.html`, because that page is treated as a distinct route.
2. Unknown browser paths should still resolve to `index.html`, because the UI behaves like a lightweight SPA.

That design prevents the common “refresh causes 404” problem in static deployments.

## Component Table

| Component | Port | Role |
|---|---|---|
| Frontend | 80 | Serves the operator UI and static assets |
| FastAPI API | 8000 | JSON API and model-serving host |
| FastAPI Metrics | 8001 | Prometheus scrape endpoint |
| MLflow | 5000 | Experiment tracking and registry |
| Airflow | 8080 | Orchestration UI |
| Prometheus | 9090 | Metrics scraping and query layer |
| Grafana | 3000 | Dashboards and visual monitoring |

## Persistence and Recovery

### Survives container restarts

- `data/raw/` because it holds persisted downloads.
- `data/processed/` because it holds transformed data.
- `data/baselines/` because it stores reference artifacts.
- `data/simulation/` because it stores prediction history.
- `models/` because it stores trained model assets.
- `mlruns/` because it stores MLflow runs.
- `logs/` because it stores runtime logs.
- Grafana dashboard state because the data volume is persisted.

### Does not survive container recreation cleanly

- In-memory API state.
- Temporary Python objects inside the prediction process.
- Temporary Prometheus scrape cache.
- Runtime-only orchestration state outside mounted folders.

## Viva-Style Summary

If I had to explain this project in a viva, I would say the system is a regime-detection pipeline with a clean split between data engineering, model training, model serving, and browser presentation. The Python modules own the logic, the filesystem owns the persisted artifacts, the API owns the runtime contract, and Nginx owns static delivery.

The recent changes make the system more reliable in practice because they remove a dependency-shape assumption from yfinance and make the frontend routing more deployable. In other words, the model logic did not change, but the infrastructure around it became more robust.

That is the main lesson of the architecture: the best pipeline is not only the one that produces good predictions, but the one that keeps working when libraries, containers, and deployment environments change around it.