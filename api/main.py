"""FastAPI application for the Market Regime Detection system."""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    DriftResponse,
    DriftFeatureReport,
    GroundTruthInput,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RetrainResponse,
    SingleRegimeResult,
)
from src import data_ingestion, drift_monitor, monitoring, predict, retraining_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PREDICTION_LOG = PROJECT_ROOT / "data" / "simulation" / "prediction_log.csv"
GROUND_TRUTH_LOG = PROJECT_ROOT / "data" / "baselines" / "ground_truth_log.csv"
TRAINING_REPORT = PROJECT_ROOT / "data" / "baselines" / "training_report.json"
SERVICE_TARGETS = {
    "mlflow": "http://mlflow:5000/health",
    "airflow": "http://airflow:8080/health",
    "prometheus": "http://prometheus:9090/-/ready",
    "grafana": "http://grafana:3000/api/health",
}

START_TIME = time.time()
SERVICE_VERSION = "1.0.0"

logger = logging.getLogger(__name__)
REQUEST_HISTORY = deque(maxlen=100)

app = FastAPI(title="Market Regime Detection API", version=SERVICE_VERSION, docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _ensure_prediction_log() -> None:
    """Create the simulation prediction log if it does not yet exist."""

    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    expected_header = [
        "timestamp",
        "simulation_date",
        "ticker",
        "regime_type",
        "predicted_label",
        "confidence",
        "inference_latency_ms",
        "features_snapshot",
    ]

    if not PREDICTION_LOG.exists():
        with PREDICTION_LOG.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(expected_header)
        return

    with PREDICTION_LOG.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().split(",")

    if first_line == expected_header:
        return

    try:
        with PREDICTION_LOG.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        with PREDICTION_LOG.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_header)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "simulation_date": row.get("simulation_date", ""),
                        "ticker": row.get("ticker", ""),
                        "regime_type": row.get("regime_type", ""),
                        "predicted_label": row.get("predicted_label", ""),
                        "confidence": row.get("confidence", ""),
                        "inference_latency_ms": row.get("inference_latency_ms", ""),
                        "features_snapshot": row.get("features_snapshot", "{}"),
                    }
                )
    except Exception as exc:
        logger.warning("Unable to upgrade prediction log schema cleanly: %s", exc)
        with PREDICTION_LOG.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def _append_prediction_log_row(timestamp: str, ticker: str, regime_type: str, predicted_label: str, confidence: float, features_snapshot: dict[str, Any], simulation_date: str = "", inference_latency_ms: float = 0.0) -> None:
    """Append a single prediction row to the live simulation log."""

    _ensure_prediction_log()
    with PREDICTION_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                timestamp,
                simulation_date,
                ticker,
                regime_type,
                predicted_label,
                f"{confidence:.6f}",
                f"{inference_latency_ms:.3f}",
                json.dumps(features_snapshot, separators=(",", ":")),
            ]
        )


def _append_ground_truth_row(payload: GroundTruthInput) -> None:
    """Append a feedback row for later evaluation of model predictions."""

    GROUND_TRUTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not GROUND_TRUTH_LOG.exists():
        GROUND_TRUTH_LOG.write_text("date,ticker,regime_type,actual_label\n", encoding="utf-8")
    with GROUND_TRUTH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{payload.date},{payload.ticker},{payload.regime_type},{payload.actual_label}\n")


def _load_models_loaded_state() -> dict[str, bool]:
    """Return a boolean map of loaded regime models."""

    registry = predict.ModelRegistry.get_instance()
    return {regime: regime in registry.models for regime in ["trend", "vol", "bull_bear"]}


def _pipeline_status_payload() -> dict[str, Any]:
    """Build a compact snapshot of pipeline health for the frontend."""

    status: dict[str, Any] = {
        "last_ingestion_time": None,
        "last_training_time": None,
        "last_drift_check": None,
        "model_versions": {},
        "model_metrics": {},
        "ground_truth_rows": 0,
    }

    if TRAINING_REPORT.exists():
        try:
            with TRAINING_REPORT.open("r", encoding="utf-8") as handle:
                report = json.load(handle) or {}
            status["last_training_time"] = report.get("timestamp")
            status["model_versions"] = {item.get("regime_type", "unknown"): item.get("model_version") for item in report.get("results", [])}
            status["model_metrics"] = {
                item.get("regime_type", "unknown"): {
                    "test_f1": item.get("test_f1"),
                    "test_accuracy": item.get("test_accuracy"),
                }
                for item in report.get("results", [])
            }
            status["last_ingestion_time"] = report.get("timestamp")
        except Exception as exc:
            logger.warning("Unable to read training report: %s", exc)

    raw_candidates = [
        PROJECT_ROOT / "data" / "raw" / "SPY_train.csv",
        PROJECT_ROOT / "data" / "raw" / "SPY_sim.csv",
    ]
    for candidate in raw_candidates:
        if candidate.exists():
            try:
                status["last_ingestion_time"] = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc).isoformat()
                break
            except Exception:
                continue

    if GROUND_TRUTH_LOG.exists():
        try:
            with GROUND_TRUTH_LOG.open("r", encoding="utf-8") as handle:
                status["ground_truth_rows"] = max(sum(1 for _ in handle) - 1, 0)
        except Exception as exc:
            logger.warning("Unable to read ground truth log: %s", exc)

    if PREDICTION_LOG.exists():
        try:
            status["last_drift_check"] = datetime.fromtimestamp(PREDICTION_LOG.stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            status["last_drift_check"] = None

    return status


def _prediction_history_payload(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent prediction rows for the frontend timeline."""

    if not PREDICTION_LOG.exists():
        return []

    try:
        with PREDICTION_LOG.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        return rows[-limit:]
    except Exception as exc:
        logger.warning("Unable to read prediction history: %s", exc)
        return []


def _check_service_health(service_name: str) -> dict[str, Any]:
    """Probe a supporting container from inside the Docker network."""

    if service_name not in SERVICE_TARGETS:
        raise KeyError(f"Unknown service: {service_name}")

    url = SERVICE_TARGETS[service_name]
    try:
        response = requests.get(url, timeout=3)
        return {
            "service": service_name,
            "reachable": response.ok,
            "status_code": response.status_code,
            "url": url,
        }
    except Exception as exc:
        return {
            "service": service_name,
            "reachable": False,
            "status_code": None,
            "url": url,
            "error": str(exc),
        }


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize metrics and the model registry when the API starts."""

    monitoring.start_metrics_server(port=8001)
    registry = predict.ModelRegistry.get_instance()
    monitoring.MODELS_LOADED.set(3 if registry.is_ready() else len(registry.models))
    logger.info("[API] Startup complete. Models: %s. Metrics: port 8001", list(registry.models.keys()))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request metrics and update the rolling error rate gauge."""

    start = time.time()
    response = await call_next(request)
    monitoring.record_api_request(request.url.path, request.method, response.status_code)
    elapsed_ms = (time.time() - start) * 1000.0
    try:
        REQUEST_HISTORY.append(int(response.status_code))
        errors = sum(1 for code in REQUEST_HISTORY if code >= 400)
        monitoring.set_error_rate_percent((errors / max(len(REQUEST_HISTORY), 1)) * 100.0)
    except Exception:
        pass
    logger.info("[API] %s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return a simple health summary for the service and model registry."""

    try:
        registry = predict.ModelRegistry.get_instance()
        models_loaded = _load_models_loaded_state()
        if all(models_loaded.values()):
            status = "ok"
        elif any(models_loaded.values()):
            status = "degraded"
        else:
            status = "down"
        uptime_seconds = time.time() - START_TIME
        logger.info("[API] health check -> %s", status)
        return HealthResponse(status=status, models_loaded=models_loaded, uptime_seconds=uptime_seconds, version=SERVICE_VERSION)
    except Exception as exc:
        logger.exception("[API] health check failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """Return readiness for Docker health checks."""

    try:
        registry = predict.ModelRegistry.get_instance()
        if registry.is_ready():
            return {"ready": True}
        raise HTTPException(status_code=503, detail={"ready": False, "reason": "models not loaded"})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[API] readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail={"ready": False, "reason": "models not loaded"}) from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Generate predictions for one ticker across the requested regime types."""

    start = time.time()
    try:
        registry = predict.ModelRegistry.get_instance()
        if not registry.is_ready():
            raise HTTPException(status_code=500, detail="models not loaded")

        results: dict[str, SingleRegimeResult] = {}
        for regime_type in payload.regime_types:
            if regime_type not in ["trend", "vol", "bull_bear"]:
                raise HTTPException(status_code=422, detail=f"Invalid regime type: {regime_type}")
            regime_result = predict.predict_regime(payload.ticker, regime_type, registry=registry, as_of_date=payload.as_of_date)
            results[regime_type] = SingleRegimeResult(
                regime_type=regime_result["regime_type"],
                predicted_label=str(regime_result["predicted_label"]),
                confidence=float(regime_result["confidence"]),
                proba_class_0=float(regime_result["proba_class_0"]),
                proba_class_1=float(regime_result["proba_class_1"]),
                inference_date=str(regime_result["inference_date"]),
                features_used=int(regime_result["features_used"]),
                inference_latency_ms=float(regime_result["inference_latency_ms"]),
            )
            monitoring.record_prediction(
                regime_type=regime_type,
                predicted_label=str(regime_result["predicted_label"]),
                ticker=payload.ticker,
                latency_ms=float(regime_result["inference_latency_ms"]),
                confidence=float(regime_result["confidence"]),
            )
            _append_prediction_log_row(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ticker=payload.ticker,
                regime_type=regime_type,
                predicted_label=str(regime_result["predicted_label"]),
                confidence=float(regime_result["confidence"]),
                features_snapshot=regime_result["features_snapshot"],
                simulation_date=payload.as_of_date or "",
                inference_latency_ms=float(regime_result["inference_latency_ms"]),
            )
            logger.info("[API] prediction %s/%s -> %s", payload.ticker, regime_type, regime_result["predicted_label"])

        total_latency_ms = (time.time() - start) * 1000.0
        logger.info("[API] predict completed for %s in %.1f ms", payload.ticker, total_latency_ms)
        return PredictionResponse(ticker=payload.ticker, timestamp=datetime.now(timezone.utc).isoformat(), results=results, total_latency_ms=total_latency_ms)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[API] prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/drift/{regime_type}", response_model=DriftResponse)
async def drift_endpoint(regime_type: str) -> DriftResponse:
    """Return the current drift report for a regime type."""

    try:
        if regime_type not in ["trend", "vol", "bull_bear"]:
            raise HTTPException(status_code=422, detail=f"Invalid regime type: {regime_type}")
        drift_reports = drift_monitor.check_drift_from_log(regime_type)
        monitoring.record_drift(regime_type, drift_reports)
        retrain_recommended, retrain_reason = drift_monitor.should_retrain(regime_type)
        any_drift_detected = any(item["drift_detected"] for item in drift_reports)
        logger.info("[API] drift check for %s -> %s", regime_type, retrain_reason)
        return DriftResponse(
            regime_type=regime_type,
            n_recent_samples=len(drift_reports),
            drift_reports=[DriftFeatureReport(**report) for report in drift_reports],
            any_drift_detected=any_drift_detected,
            retrain_recommended=retrain_recommended,
            retrain_reason=retrain_reason,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[API] drift endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_endpoint(triggered_by: str = "manual") -> RetrainResponse:
    """Start a non-blocking retraining job."""

    try:
        monitoring.RETRAINING_TRIGGERED_COUNTER.labels(trigger_reason=triggered_by).inc()
        manager = retraining_manager.RetrainingManager()
        async_result = manager.run_retrain_async(triggered_by=triggered_by)
        logger.info("[API] retrain triggered by %s -> %s", triggered_by, async_result)
        return RetrainResponse(status=async_result["status"], triggered_by=triggered_by, message="Retraining started in background")
    except Exception as exc:
        logger.exception("[API] retrain endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_ingestion_job() -> dict[str, Any]:
    """Run the ingestion workflow used by the pipeline screen."""

    training_result = data_ingestion.download_training_data()
    simulation_result = data_ingestion.download_simulation_data()
    baseline_result = data_ingestion.compute_baseline_stats()
    return {
        "training_files": len(training_result["validations"]),
        "simulation_rows": simulation_result["total_rows"],
        "baseline_path": str(baseline_result["path"]),
    }


@app.post("/pipeline/ingest")
async def pipeline_ingest(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Trigger the ingestion workflow from the pipeline screen."""

    try:
        background_tasks.add_task(_run_ingestion_job)
        logger.info("[API] ingestion triggered from pipeline screen")
        return {"status": "started", "message": "Ingestion started in background"}
    except Exception as exc:
        logger.exception("[API] pipeline ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ground-truth")
async def ground_truth_endpoint(payload: GroundTruthInput) -> dict[str, Any]:
    """Record delayed ground-truth labels for later model evaluation."""

    try:
        _append_ground_truth_row(payload)
        logger.info("[API] recorded ground truth for %s/%s", payload.ticker, payload.regime_type)
        return {"status": "recorded", "date": payload.date, "actual_label": payload.actual_label}
    except Exception as exc:
        logger.exception("[API] ground-truth endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/pipeline-status")
async def pipeline_status_endpoint() -> dict[str, Any]:
    """Return the current status of the pipeline components for the frontend."""

    try:
        return _pipeline_status_payload()
    except Exception as exc:
        logger.exception("[API] pipeline status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/prediction-history")
async def prediction_history_endpoint(limit: int = 50) -> dict[str, Any]:
    """Return recent simulated predictions for charts and the live feed."""

    try:
        history = _prediction_history_payload(limit=max(1, min(limit, 250)))
        return {"count": len(history), "rows": history}
    except Exception as exc:
        logger.exception("[API] prediction history failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/service-health/{service_name}")
async def service_health_endpoint(service_name: str) -> dict[str, Any]:
    """Return the health of a supporting service reachable from the API container."""

    try:
        return _check_service_health(service_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    """Serve the main frontend prediction UI."""

    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline() -> FileResponse:
    """Serve the pipeline visualization screen."""

    return FileResponse(FRONTEND_DIR / "pipeline.html")