# Test Plan

## Objectives

- Verify ingestion produces usable OHLCV CSVs and baseline statistics.
- Verify feature engineering outputs are numeric, leak-free, and non-empty.
- Verify prediction helpers return the expected contract for all regimes.
- Verify the FastAPI service exposes the expected operational endpoints.
- Verify the Dockerized deployment can run the full stack without manual setup beyond Grafana dashboard import.

## Test Cases

| ID | Description | Input | Expected Output | Acceptance Criteria |
|---|---|---|---|---|
| ING-01 | Download a ticker into CSV | Synthetic SPY OHLCV response | DataFrame plus validation payload | File is written and validation status is `PASS` |
| ING-02 | Empty frame validation | Empty DataFrame | Validation failure payload | Status is `FAIL` |
| ING-03 | Baseline stats generation | Raw SPY training CSV | `feature_baselines.json` | File exists and includes close-price statistics |
| FEAT-01 | RSI boundedness | Rising price series | RSI values | All RSI values are between 0 and 100 |
| FEAT-02 | RSI overbought behavior | Strong uptrend price series | RSI values | Final RSI is greater than 70 |
| FEAT-03 | Trend feature leakage guard | Synthetic market frame | Trend features DataFrame | Hurst/autocorr label helpers are absent from output |
| FEAT-04 | NaN cleanup | Synthetic market frame | Trend features DataFrame | Output contains no NaN values |
| PRED-01 | Prediction contract | Monkeypatched registry and features | Prediction dictionary | Required keys are present |
| PRED-02 | Confidence range | Monkeypatched registry and features | Prediction dictionary | Confidence is between 0 and 1 |
| PRED-03 | Label validity | Monkeypatched registry and features | Prediction dictionary | Predicted label belongs to the expected class set |
| API-01 | Health endpoint | `GET /health` | 200 response | `status`, `models_loaded`, and `uptime_seconds` are present |
| API-02 | Readiness endpoint | `GET /ready` | 200 or 503 response | Endpoint exists and responds deterministically |
| API-03 | Prediction endpoint | `POST /predict` | 200 response | `results` contains `trend`, `vol`, and `bull_bear` |
| API-04 | Invalid ticker handling | `POST /predict` with invalid ticker | 500 response | API returns a server error when inference fails |

## Execution Instructions

Run the automated suite with:

```bash
pip install pytest && pytest tests/ -v --tb=short
```

If you are validating inside the Docker stack, run:

```bash
docker-compose up --build
```

Then open the API docs and frontend, and confirm the health, prediction, drift, retraining, and pipeline screens work.

## Test Report Template

| ID | Result | Notes |
|---|---|---|
| ING-01 |  |  |
| ING-02 |  |  |
| ING-03 |  |  |
| FEAT-01 |  |  |
| FEAT-02 |  |  |
| FEAT-03 |  |  |
| FEAT-04 |  |  |
| PRED-01 |  |  |
| PRED-02 |  |  |
| PRED-03 |  |  |
| API-01 |  |  |
| API-02 |  |  |
| API-03 |  |  |
| API-04 |  |  |

Fill the notes column with any deviations, failing assertions, or follow-up actions.
