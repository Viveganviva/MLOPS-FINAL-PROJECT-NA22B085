"""Prometheus metrics and background system collection for the API layer."""

from __future__ import annotations

import logging
import threading
import time

import psutil

try:  # pragma: no cover - optional dependency guard
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
except Exception:  # pragma: no cover - fallback when prometheus_client is unavailable
    class _NoOpMetric:
        def labels(self, **_: str):
            return self

        def inc(self, *_: float, **__: float) -> None:
            return None

        def observe(self, *_: float, **__: float) -> None:
            return None

        def set(self, *_: float, **__: float) -> None:
            return None


    def _factory(*_: object, **__: object) -> _NoOpMetric:
        return _NoOpMetric()


    Counter = Histogram = Gauge = Summary = _factory  # type: ignore[assignment]

    def start_http_server(port: int) -> None:  # type: ignore[override]
        logger.warning("prometheus_client is unavailable; metrics endpoint on port %s will be a no-op", port)


logger = logging.getLogger(__name__)


PREDICTION_COUNTER = Counter(
    "regime_predictions_total",
    "Total predictions made, labeled by regime type and predicted class",
    ["regime_type", "predicted_label", "ticker"],
)

PREDICTION_LATENCY_MS = Histogram(
    "regime_prediction_latency_milliseconds",
    "End-to-end inference latency in milliseconds (includes feature computation)",
    ["regime_type"],
    buckets=[10, 25, 50, 100, 200, 500, 1000, 2000],
)

MODEL_CONFIDENCE = Gauge(
    "regime_model_confidence_score",
    "Confidence (max class probability) of the most recent prediction",
    ["regime_type", "ticker"],
)

LAST_PREDICTION_UNIX_TS = Gauge(
    "regime_last_prediction_timestamp_seconds",
    "Unix timestamp of the last prediction made",
    ["regime_type"],
)

DATA_DRIFT_KL_SCORE = Gauge(
    "regime_data_drift_kl_score",
    "KL divergence score for each monitored feature vs training baseline",
    ["regime_type", "feature_name"],
)

DRIFT_DETECTED_FLAG = Gauge(
    "regime_drift_detected",
    "1 if drift detected for this regime type, 0 otherwise",
    ["regime_type"],
)

RETRAINING_TRIGGERED_COUNTER = Counter(
    "regime_retraining_triggered_total",
    "Total number of times model retraining was triggered",
    ["trigger_reason"],
)

RETRAINING_DURATION_SECONDS = Gauge(
    "regime_retraining_duration_seconds",
    "Duration of the last completed retraining run in seconds",
)

CPU_USAGE_PERCENT = Gauge(
    "system_cpu_usage_percent",
    "Current CPU usage percentage of the API process",
)

MEMORY_USAGE_MB = Gauge(
    "system_memory_usage_mb",
    "Current RAM usage of the API process in megabytes",
)

MEMORY_USAGE_PERCENT = Gauge(
    "system_memory_usage_percent",
    "System-wide memory usage percentage",
)

DISK_USAGE_PERCENT = Gauge(
    "system_disk_usage_percent",
    "Disk usage percentage of the working directory volume",
)

API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total HTTP requests received",
    ["endpoint", "method", "status_code"],
)

API_ERROR_RATE = Gauge(
    "api_error_rate_percent",
    "Rolling error rate over last 100 requests (percentage)",
)

MODELS_LOADED = Gauge(
    "regime_models_loaded_count",
    "Number of regime models currently loaded and ready",
)


class SystemMetricsCollector:
    """
    Runs in a background thread and updates system-level Prometheus gauges
    every 15 seconds. Keeps the API endpoint non-blocking.
    """

    def __init__(self, interval_seconds: int = 15):
        self.interval = interval_seconds
        self.process = psutil.Process()
        self._thread: threading.Thread | None = None
        self._running = False

    def collect(self) -> None:
        """Refresh the system gauges with the current host and process metrics."""

        CPU_USAGE_PERCENT.set(self.process.cpu_percent(interval=1))
        mem = self.process.memory_info()
        MEMORY_USAGE_MB.set(mem.rss / 1024 / 1024)
        MEMORY_USAGE_PERCENT.set(psutil.virtual_memory().percent)
        DISK_USAGE_PERCENT.set(psutil.disk_usage(".").percent)

    def start(self) -> None:
        """Start the background metrics thread if it is not already running."""

        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[Monitoring] System metrics collector started (interval: %ss)", self.interval)

    def _run(self) -> None:
        """Collect system metrics on a fixed cadence until stopped."""

        while self._running:
            try:
                self.collect()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                logger.warning("[Monitoring] System metric collection failed: %s", exc)
            time.sleep(self.interval)

    def stop(self) -> None:
        """Stop the background collection loop."""

        self._running = False


SYSTEM_COLLECTOR = SystemMetricsCollector()


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus HTTP exporter and the system collector."""

    start_http_server(port)
    SYSTEM_COLLECTOR.start()
    logger.info("[Monitoring] Prometheus metrics available at http://0.0.0.0:%s/metrics", port)


def record_prediction(regime_type: str, predicted_label: str, ticker: str, latency_ms: float, confidence: float) -> None:
    """Call this in the API after every successful prediction."""

    PREDICTION_COUNTER.labels(regime_type=regime_type, predicted_label=predicted_label, ticker=ticker).inc()
    PREDICTION_LATENCY_MS.labels(regime_type=regime_type).observe(latency_ms)
    MODEL_CONFIDENCE.labels(regime_type=regime_type, ticker=ticker).set(confidence)
    LAST_PREDICTION_UNIX_TS.labels(regime_type=regime_type).set(time.time())


def record_drift(regime_type: str, drift_results: list[dict[str, object]]) -> None:
    """Call this after a drift check to update Grafana gauges."""

    any_drift = False
    for drift_result in drift_results:
        feature_name = str(drift_result["feature_name"])
        DATA_DRIFT_KL_SCORE.labels(regime_type=regime_type, feature_name=feature_name).set(float(drift_result["kl_score"]))
        if bool(drift_result["drift_detected"]):
            any_drift = True
    DRIFT_DETECTED_FLAG.labels(regime_type=regime_type).set(1 if any_drift else 0)


def record_api_request(endpoint: str, method: str, status_code: int | str) -> None:
    """Call this as middleware in FastAPI."""

    API_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code=str(status_code)).inc()


def set_error_rate_percent(value: float) -> None:
    """Update the rolling API error-rate gauge."""

    API_ERROR_RATE.set(value)