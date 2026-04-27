"""Drift monitoring utilities for live regime predictions."""

from __future__ import annotations

from statistics import mean
from typing import Mapping


def detect_drift(reference: Mapping[str, float], current: Mapping[str, float], threshold: float = 0.15) -> dict[str, float | bool]:
    """Compute a small, transparent drift signal from feature summaries."""

    shared_keys = sorted(set(reference) & set(current))
    if not shared_keys:
        raise ValueError("reference and current must share at least one feature")

    deltas = [abs(current[key] - reference[key]) for key in shared_keys]
    drift_score = mean(deltas)
    return {
        "drift_score": drift_score,
        "threshold": threshold,
        "is_drifted": drift_score >= threshold,
    }