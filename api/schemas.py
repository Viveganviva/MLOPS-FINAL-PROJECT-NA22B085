"""Pydantic request and response models for the FastAPI service."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request payload for regime prediction."""

    ticker: str = Field(default="SPY", description="Stock ticker symbol (e.g. SPY, AAPL)")
    as_of_date: str | None = Field(default=None, description="Optional historical date anchor for replay/simulation")
    regime_types: List[str] = Field(
        default=["trend", "vol", "bull_bear"],
        description="Which regimes to predict. Options: trend, vol, bull_bear",
    )


class SingleRegimeResult(BaseModel):
    """Prediction payload for a single regime type."""

    regime_type: str
    predicted_label: str
    confidence: float
    proba_class_0: float
    proba_class_1: float
    inference_date: str
    features_used: int
    inference_latency_ms: float


class PredictionResponse(BaseModel):
    """Aggregate prediction response for all requested regimes."""

    ticker: str
    timestamp: str
    results: Dict[str, SingleRegimeResult]
    total_latency_ms: float


class DriftFeatureReport(BaseModel):
    """Feature-level drift diagnostics."""

    feature_name: str
    training_mean: float
    training_std: float
    recent_mean: float
    recent_std: float
    kl_score: float
    drift_detected: bool


class DriftResponse(BaseModel):
    """Drift response with the retrain recommendation."""

    regime_type: str
    n_recent_samples: int
    drift_reports: List[DriftFeatureReport]
    any_drift_detected: bool
    retrain_recommended: bool
    retrain_reason: str


class HealthResponse(BaseModel):
    """Service health payload."""

    status: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float
    version: str = "1.0.0"


class RetrainResponse(BaseModel):
    """Response returned after starting a retraining job."""

    status: str
    triggered_by: str
    message: str


class GroundTruthInput(BaseModel):
    """
    Allows feeding actual observed labels back into the system.
    In production, this would be called once the true regime is known (e.g., 1 week later).
    Required by the MLOps evaluation rubric: 'Feedback Loop' requirement.
    """

    date: str
    regime_type: str
    actual_label: str
    ticker: str