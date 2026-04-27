"""FastAPI application for regime inference."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schemas import HealthResponse, PredictionRequest, PredictionResponse

try:  # pragma: no cover - optional dependency guard for scaffold imports
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - handled gracefully for the scaffold
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

from src.predict import predict_regime


SERVICE_NAME = "market-regime-detection-api"


def build_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI is required to run the API service.")

    app = FastAPI(title="Market Regime Detection API", version="0.1.0")

    @app.get("/health", response_model=None)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=SERVICE_NAME)

    @app.post("/predict", response_model=None)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        if not payload.features:
            raise HTTPException(status_code=400, detail="features cannot be empty")

        prediction = predict_regime(payload.features, regime_type=payload.regime_type)
        return PredictionResponse(
            timestamp=datetime.now(timezone.utc),
            ticker=payload.ticker,
            regime_type=prediction["regime_type"],
            predicted_label=prediction["predicted_label"],
            confidence=float(prediction["confidence"]),
            features_snapshot=dict(prediction["features_snapshot"]),
            metadata={"service": SERVICE_NAME},
        )

    return app


app = build_app() if FastAPI is not None else None