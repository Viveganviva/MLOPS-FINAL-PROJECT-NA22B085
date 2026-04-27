"""Retraining decision logic for the production pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrainingPolicy:
    min_predictions_before_retrain: int = 100
    retrain_cooldown_days: int = 7
    drift_threshold: float = 0.15


def should_retrain(prediction_count: int, drift_score: float, policy: RetrainingPolicy | None = None) -> bool:
    """Return whether the retraining gate should open."""

    active_policy = policy or RetrainingPolicy()
    if prediction_count < active_policy.min_predictions_before_retrain:
        return False
    return drift_score >= active_policy.drift_threshold