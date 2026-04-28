# Architecture Diagram

## ASCII Diagram

```text
                         +----------------------+
                         |      Browser UI      |
                         | index.html / pipeline.html
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |     FastAPI API      |
                         | api/main.py          |
                         | /health /predict ...  |
                         +----+-----------+-----+
                              |           |
                              |           +---------------------+
                              |                                 |
                              v                                 v
                 +------------------------+           +----------------------+
                 | Prediction / Drift Logs |           | Prometheus Metrics   |
                 | data/simulation         |           | port 8001 -> 9090    |
                 +-----------+------------+           +----------+-----------+
                             |                                   |
                             v                                   v
                 +------------------------+           +----------------------+
                 | src/predict.py         |           | Grafana Dashboard    |
                 | src/drift_monitor.py    |           | regime_dashboard.json|
                 +-----------+------------+           +----------------------+
                             |
                             v
                 +------------------------+
                 | Shared Feature Logic    |
                 | src/feature_engineering |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | Raw CSVs / Processed    |
                 | data/raw               |
                 | data/processed         |
                 | data/baselines         |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | Training / MLflow      |
                 | src/train.py           |
                 | models/ mlruns/        |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | Airflow DAG Scaffold   |
                 | airflow_dags/          |
                 +------------------------+
```

## Component Table

| Component | Port | Role |
|---|---|---|
| Frontend | 80 | Serves the operator UI |
| FastAPI API | 8000 | JSON API and static route host |
| FastAPI Metrics | 8001 | Prometheus scrape endpoint |
| MLflow | 5000 | Experiment tracking and registry |
| Airflow | 8080 | Orchestration UI |
| Prometheus | 9090 | Metrics scraping and query layer |
| Grafana | 3000 | Dashboards and visual monitoring |

## Network Flow

1. The browser opens the static frontend on port 80.
2. The frontend calls the API on port 8000.
3. The API loads model artifacts from `models/` and writes prediction history to `data/simulation/`.
4. The API exposes metrics on port 8001.
5. Prometheus scrapes the metrics endpoint.
6. Grafana reads from Prometheus.
7. MLflow stores experiment runs and artifacts.
8. Airflow provides orchestration for pipeline jobs.

## Data Persistence

### Survives container restarts

- `data/raw/` because it is mounted from the host.
- `data/processed/` because it is mounted from the host.
- `data/baselines/` because it is mounted from the host.
- `data/simulation/` because it is mounted from the host.
- `models/` because it is mounted from the host.
- `mlruns/` because it is mounted from the host.
- `logs/` because it is mounted from the host.
- Grafana dashboard state because the `grafana_data` volume is persisted.

### Does not survive container recreation cleanly

- In-memory API state.
- Background counters inside Python processes.
- Temporary Prometheus scrape cache.
- Airflow runtime state outside the mounted folders.

## Operational Note

The browser should talk to the API, not directly to Prometheus, Grafana, or Airflow. The API already provides the service-health proxy used by the pipeline screen, which avoids cross-origin problems and keeps the UI simple.