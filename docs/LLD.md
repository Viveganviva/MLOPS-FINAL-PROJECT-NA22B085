# Low-Level Design

## API Endpoint Specification

| Endpoint | Method | Input | Output | Auth |
|---|---|---|---|---|
| /health | GET | None | HealthResponse | None |
| /ready | GET | None | 200 or 503 | None |
| /predict | POST | PredictionRequest | PredictionResponse | None |
| /drift/{regime} | GET | regime_type (path) | DriftResponse | None |
| /retrain | POST | triggered_by (query) | RetrainResponse | None |
| /ground-truth | POST | GroundTruthInput | dict | None |
| /pipeline-status | GET | None | dict | None |
| /prediction-history | GET | limit query param | dict | None |
| /service-health/{service_name} | GET | service_name (path) | dict | None |

## Schema Reference

### PredictionRequest

```json
{
	"ticker": "SPY",
	"as_of_date": "2024-01-02",
	"regime_types": ["trend", "vol", "bull_bear"]
}
```

### SingleRegimeResult

```json
{
	"regime_type": "trend",
	"predicted_label": "Trending",
	"confidence": 0.91,
	"proba_class_0": 0.09,
	"proba_class_1": 0.91,
	"inference_date": "2024-01-02",
	"features_used": 42,
	"inference_latency_ms": 12.3
}
```

### PredictionResponse

```json
{
	"ticker": "SPY",
	"timestamp": "2024-01-02T12:34:56Z",
	"results": {
		"trend": { "regime_type": "trend", "predicted_label": "Trending", "confidence": 0.91, "proba_class_0": 0.09, "proba_class_1": 0.91, "inference_date": "2024-01-02", "features_used": 42, "inference_latency_ms": 12.3 },
		"vol": { "regime_type": "vol", "predicted_label": "LowVol", "confidence": 0.88, "proba_class_0": 0.12, "proba_class_1": 0.88, "inference_date": "2024-01-02", "features_used": 39, "inference_latency_ms": 10.1 },
		"bull_bear": { "regime_type": "bull_bear", "predicted_label": "Bull", "confidence": 0.93, "proba_class_0": 0.07, "proba_class_1": 0.93, "inference_date": "2024-01-02", "features_used": 44, "inference_latency_ms": 14.2 }
	},
	"total_latency_ms": 36.6
}
```

### DriftFeatureReport

```json
{
	"feature_name": "Close",
	"training_mean": 100.1,
	"training_std": 2.4,
	"recent_mean": 101.0,
	"recent_std": 2.9,
	"kl_score": 0.05,
	"drift_detected": false
}
```

### DriftResponse

```json
{
	"regime_type": "trend",
	"n_recent_samples": 50,
	"drift_reports": [
		{
			"feature_name": "Close",
			"training_mean": 100.1,
			"training_std": 2.4,
			"recent_mean": 101.0,
			"recent_std": 2.9,
			"kl_score": 0.05,
			"drift_detected": false
		}
	],
	"any_drift_detected": false,
	"retrain_recommended": false,
	"retrain_reason": "No significant drift detected"
}
```

### HealthResponse

```json
{
	"status": "ok",
	"models_loaded": { "trend": true, "vol": true, "bull_bear": true },
	"uptime_seconds": 124.3,
	"version": "1.0.0"
}
```

### RetrainResponse

```json
{
	"status": "started",
	"triggered_by": "manual",
	"message": "Retraining started in background"
}
```

### GroundTruthInput

```json
{
	"date": "2024-01-02",
	"regime_type": "trend",
	"actual_label": "Trending",
	"ticker": "SPY"
}
```

## Endpoint Behavior

### /health

1. Loads the model registry.
2. Checks whether all three regime models are in memory.
3. Returns `ok`, `degraded`, or `down` plus uptime and version.

### /ready

1. Checks whether the registry loaded successfully.
2. Returns 200 when the models are available.
3. Returns 503 when the registry is not ready.

### /predict

1. Validates the incoming request with Pydantic.
2. Loads the registry if needed.
3. Downloads recent market data through `src/predict.py`.
4. Builds regime-specific features using the shared feature-engineering module.
5. Scales the most recent valid row.
6. Calls the trained model for each requested regime.
7. Records metrics, appends to the prediction log, and returns the full response.

### /drift/{regime}

1. Reads recent prediction rows for the requested regime.
2. Parses stored feature snapshots.
3. Compares recent statistics to the baseline file.
4. Returns feature-level drift reports and a retrain recommendation.

### /retrain

1. Increments the retraining metric counter.
2. Starts the retraining manager in a background thread.
3. Returns immediately so the API stays responsive.

### /ground-truth

1. Writes the delayed label into `data/baselines/ground_truth_log.csv`.
2. Leaves a paper trail for later evaluation.

### /pipeline-status

1. Reads `training_report.json` and the data folders.
2. Summarizes last ingestion, last training, model versions, metrics, and ground-truth count.
3. Returns a compact JSON object for the pipeline UI.

### /prediction-history

1. Reads `data/simulation/prediction_log.csv`.
2. Returns the latest rows for the chart and live feed.

### /service-health/{service_name}

1. Probes supporting services from inside the Docker network.
2. Returns reachability, HTTP status, and target URL.

## Module Dependencies

```text
api/main.py
	-> api/schemas.py
	-> src/data_ingestion.py
	-> src/drift_monitor.py
	-> src/monitoring.py
	-> src/predict.py
	-> src/retraining_manager.py

src/predict.py
	-> src/feature_engineering.py
	-> models/

src/retraining_manager.py
	-> src/data_ingestion.py
	-> src/feature_engineering.py
	-> src/train.py

src/train.py
	-> data/processed/
	-> models/
	-> mlflow

src/drift_monitor.py
	-> data/baselines/feature_baselines.json
	-> data/simulation/prediction_log.csv
	-> data/baselines/training_report.json

src/save_dvc_metrics.py
	-> data/processed/
	-> models/
	-> data/baselines/training_report.json
```

## Error Codes

| Code | Meaning |
|---|---|
| 200 | Request succeeded |
| 400 | Bad request payload or missing input |
| 404 | Unknown service name for `/service-health/{service_name}` |
| 422 | Invalid regime type or validation error |
| 500 | Internal failure during prediction, drift, ingestion, or write operations |
| 503 | Models not loaded or API not ready |

## Implementation Notes

- The API is intentionally thin. It delegates the real work to the modules in `src/`.
- The same feature builders are used during training and inference so the feature columns stay aligned.
- Prediction history is stored locally so the UI and drift monitor have a stable source of truth.
- Ground-truth logging is append-only to preserve the feedback trail.