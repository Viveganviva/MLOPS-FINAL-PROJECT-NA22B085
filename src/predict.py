"""Prediction helpers for regime inference."""

from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def predict_regime(features: dict[str, float], regime_type: str = "trend") -> dict[str, Any]:
    """Return a deterministic placeholder response until trained artifacts are wired in."""

    if not features:
        raise ValueError("features must contain at least one numeric signal")

    confidence = 0.5
    label = "unknown"
    LOGGER.info("Prediction scaffold invoked for regime_type=%s with %d features", regime_type, len(features))
    return {
        "regime_type": regime_type,
        "predicted_label": label,
        "confidence": confidence,
        "features_snapshot": features,
    }