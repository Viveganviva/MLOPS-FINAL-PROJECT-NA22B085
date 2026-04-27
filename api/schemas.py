"""Request and response schemas for the prediction API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class PredictionRequest:
    ticker: str
    regime_type: str = "trend"
    features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionResponse:
    timestamp: datetime
    ticker: str
    regime_type: str
    predicted_label: str
    confidence: float
    features_snapshot: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HealthResponse:
    status: str
    service: str