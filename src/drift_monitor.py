"""Data drift detection for live regime predictions and retraining decisions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
BASELINE_PATH = PROJECT_ROOT / "data" / "baselines" / "feature_baselines.json"
PREDICTION_LOG_PATH = PROJECT_ROOT / "data" / "simulation" / "prediction_log.csv"
TRAINING_REPORT_PATH = PROJECT_ROOT / "data" / "baselines" / "training_report.json"


def _load_params() -> dict[str, Any]:
    """Load the shared parameter file for drift thresholds and cooldowns."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    with PARAMS_PATH.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")
    return params


PARAMS = _load_params()


logger = logging.getLogger(__name__)


def load_baselines(regime_type: str) -> dict[str, dict[str, float]]:
    """Load the per-regime baseline statistics used for drift calculations."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline file not found at {BASELINE_PATH}. Run the ingestion and feature-engineering pipeline first."
        )

    with BASELINE_PATH.open("r", encoding="utf-8") as handle:
        baselines = json.load(handle) or {}

    if regime_type not in baselines:
        raise FileNotFoundError(f"No baselines stored for regime_type='{regime_type}'.")

    return baselines[regime_type]


def kl_divergence_gaussian(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Compute the closed-form KL divergence between two Gaussian distributions."""

    sigma1 = max(float(sigma1), 1e-9)
    sigma2 = max(float(sigma2), 1e-9)
    # This is the closed-form KL for Gaussians, which avoids sampling noise and gives a stable comparison for monitoring.
    return float(0.5 * (np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (sigma2**2) - 1.0))


def compute_drift_scores(recent_features_df: pd.DataFrame, regime_type: str) -> list[dict[str, Any]]:
    """Compute feature-level KL drift scores against the stored training baselines."""

    baselines = load_baselines(regime_type)
    threshold = float(PARAMS["monitoring"]["drift_threshold"])
    drift_results: list[dict[str, Any]] = []

    for feature_name, stats in baselines.items():
        if feature_name not in recent_features_df.columns:
            continue

        series = pd.to_numeric(recent_features_df[feature_name], errors="coerce").dropna()
        if series.empty:
            continue

        recent_mean = float(series.mean())
        recent_std = float(series.std(ddof=1) if len(series) > 1 else 0.0)
        training_mean = float(stats.get("mean", 0.0))
        training_std = float(stats.get("std", 0.0))
        kl_score = kl_divergence_gaussian(training_mean, training_std, recent_mean, recent_std if recent_std > 0 else 1e-9)
        drift_detected = kl_score > threshold

        if drift_detected:
            logger.warning(
                "[DRIFT WARNING] Feature '%s' KL divergence: %.2f (threshold: %.2f)",
                feature_name,
                kl_score,
                threshold,
            )

        drift_results.append(
            {
                "feature_name": feature_name,
                "training_mean": training_mean,
                "training_std": training_std,
                "recent_mean": recent_mean,
                "recent_std": recent_std,
                "kl_score": float(kl_score),
                "drift_detected": bool(drift_detected),
            }
        )

    return drift_results


def check_drift_from_log(regime_type: str, n_recent: int = 50) -> list[dict[str, Any]]:
    """Inspect the prediction log and compute drift from the latest feature snapshots."""

    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found at {PREDICTION_LOG_PATH}")

    log_frame = pd.read_csv(PREDICTION_LOG_PATH)
    if "features_snapshot" not in log_frame.columns:
        raise ValueError("prediction_log.csv must include a features_snapshot column")

    regime_rows = log_frame.loc[log_frame["regime_type"] == regime_type].tail(n_recent)
    if len(regime_rows) < n_recent:
        logger.warning("Only %d recent rows available for regime_type=%s; using what is available.", len(regime_rows), regime_type)

    feature_records: list[dict[str, Any]] = []
    for raw_snapshot in regime_rows["features_snapshot"].tolist():
        try:
            feature_dict = json.loads(raw_snapshot) if isinstance(raw_snapshot, str) else dict(raw_snapshot)
            numeric_snapshot = {key: value for key, value in feature_dict.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            feature_records.append(numeric_snapshot)
        except Exception as exc:
            logger.warning("Skipping malformed feature snapshot: %s", exc)

    if not feature_records:
        return []

    recent_features_df = pd.DataFrame(feature_records)
    return compute_drift_scores(recent_features_df, regime_type)


def should_retrain(regime_type: str) -> tuple[bool, str]:
    """Conservatively decide whether retraining is warranted for a regime."""

    threshold = float(PARAMS["monitoring"]["drift_threshold"])
    minimum_predictions = int(PARAMS["monitoring"]["min_predictions_before_retrain"])
    cooldown_days = int(PARAMS["monitoring"]["retrain_cooldown_days"])

    if not PREDICTION_LOG_PATH.exists():
        logger.info("[RETRAIN DECISION] %s -> False (no prediction log yet)", regime_type)
        return False, "Insufficient production data"

    log_frame = pd.read_csv(PREDICTION_LOG_PATH)
    regime_rows = log_frame.loc[log_frame["regime_type"] == regime_type]
    if len(regime_rows) < minimum_predictions:
        logger.info("[RETRAIN DECISION] %s -> False (only %d predictions)", regime_type, len(regime_rows))
        return False, "Insufficient production data"

    if TRAINING_REPORT_PATH.exists():
        with TRAINING_REPORT_PATH.open("r", encoding="utf-8") as handle:
            report = json.load(handle) or {}
        timestamp = report.get("timestamp")
        if timestamp:
            last_trained = datetime.fromisoformat(timestamp)
            if last_trained.tzinfo is None:
                last_trained = last_trained.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last_trained).days
            if age_days < cooldown_days:
                logger.info("[RETRAIN DECISION] %s -> False (cooldown active, %d days old)", regime_type, age_days)
                return False, "Cooldown active"

    drift_results = check_drift_from_log(regime_type, n_recent=min(50, len(regime_rows)))
    drifting_features = sum(1 for result in drift_results if result["drift_detected"])
    total_features = max(len(drift_results), 1)

    # We require 30% of features to be drifting (not just one) to avoid false positives from normal market noise. The cooldown prevents retraining loops.
    if drifting_features / total_features >= 0.30:
        reason = f"Drift in {drifting_features} features"
        logger.info("[RETRAIN DECISION] %s -> True (%s)", regime_type, reason)
        return True, reason

    logger.info("[RETRAIN DECISION] %s -> False (no significant drift detected)", regime_type)
    return False, "No significant drift detected"


def generate_drift_report_text(drift_results: list[dict[str, Any]]) -> str:
    """Format a drift report suitable for logs and API responses."""

    if not drift_results:
        return "DRIFT REPORT\n─────────────────────────────────────────────────\nNo feature snapshots available."

    lines = [
        "DRIFT REPORT",
        "─────────────────────────────────────────────────",
        "Feature                  KL Score   Drift?",
        "─────────────────────────────────────────────────",
    ]
    drifting = 0
    for result in drift_results:
        drift_flag = "⚠ YES" if result["drift_detected"] else "OK"
        if result["drift_detected"]:
            drifting += 1
        lines.append(f"{result['feature_name']:<24} {result['kl_score']:<10.2f} {drift_flag}")

    lines.append("─────────────────────────────────────────────────")
    lines.append(f"Drifting features: {drifting}/{len(drift_results)}")
    lines.append(f"Retrain recommended: {'YES' if drifting / max(len(drift_results), 1) >= 0.30 else 'NO'}")
    return "\n".join(lines)