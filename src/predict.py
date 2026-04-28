"""Inference engine for regime predictions."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

try:  # pragma: no cover - optional dependency guard
    import mlflow
    import mlflow.sklearn
except Exception:  # pragma: no cover - runtime fallback when MLflow is unavailable
    mlflow = None  # type: ignore[assignment]

from src import feature_engineering


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MODELS_DIR = PROJECT_ROOT / "models"


def _load_params() -> dict[str, Any]:
    """Load the shared parameter file used by the prediction engine."""

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


class ModelRegistry:
    """Singleton cache for models, scalers, and feature column order."""

    _instance: "ModelRegistry | None" = None

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Return the shared registry instance."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, Any] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.classes: dict[str, list[str]] = {}
        self.load_all()

    def load_all(self) -> None:
        """Load the artifacts for all three regime models."""

        for regime in ["trend", "vol", "bull_bear"]:
            self.models[regime] = load_model(regime)
            self.scalers[regime] = joblib.load(MODELS_DIR / f"{regime}_scaler.pkl")
            feature_path = MODELS_DIR / f"{regime}_feature_columns.json"
            if feature_path.exists():
                with feature_path.open("r", encoding="utf-8") as handle:
                    self.feature_columns[regime] = json.load(handle)
            else:
                self.feature_columns[regime] = []
            classes_path = MODELS_DIR / f"{regime}_classes.json"
            if classes_path.exists():
                with classes_path.open("r", encoding="utf-8") as handle:
                    self.classes[regime] = json.load(handle)
            else:
                self.classes[regime] = []
            logger.info("Loaded %s model, scaler, and feature schema", regime)

    def is_ready(self) -> bool:
        """Return True when all three regime models are loaded."""

        return all(regime in self.models for regime in ["trend", "vol", "bull_bear"])


def load_model(regime_type: str):
    """
    Load the production model for a given regime type.
    Tries the MLflow model registry first — this ensures we always use the version
    tagged as 'Production'. Falls back to local pkl if MLflow is unavailable.
    """

    model_name = PARAMS["mlflow"]["model_names"][regime_type]
    try:
        if mlflow is None:
            raise RuntimeError("MLflow not installed")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", PARAMS["mlflow"]["tracking_uri"]))
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        logger.info("Loaded %s model from MLflow registry (Production stage)", regime_type)
        return model
    except Exception as exc:
        logger.warning("MLflow registry unavailable (%s). Loading local pkl.", exc)
        return joblib.load(MODELS_DIR / f"{regime_type}_model.pkl")


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns into a single level when needed."""

    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = ["_".join(str(part) for part in column if part) for column in result.columns]
    return result


def fetch_recent_data(ticker: str, lookback_days: int = 300, regime_type: str | None = None, as_of_date: str | None = None) -> dict[str, pd.DataFrame]:
    """Download recent market data needed for inference."""

    if as_of_date:
        end = pd.Timestamp(as_of_date).normalize() + pd.Timedelta(days=1)
    else:
        end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=lookback_days * 2)

    tickers = [ticker]
    if regime_type == "bull_bear" or ticker.upper() == "SPY":
        tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "^VIX"]
    elif regime_type == "vol":
        tickers = [ticker, "^VIX"]

    frames: dict[str, pd.DataFrame] = {}
    for symbol in tickers:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, actions=False)
        data = _flatten_columns(data)
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        data.index = pd.to_datetime(data.index)
        frames[symbol] = data.tail(lookback_days)
        logger.info("[FETCH] %s: %d rows, latest date: %s", symbol, len(frames[symbol]), frames[symbol].index.max())

    return frames


def _prepare_feature_row(feature_frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Timestamp]:
    """Select the most recent feature row and align it to the training schema."""

    aligned = feature_frame.copy()
    if feature_columns:
        aligned = aligned.reindex(columns=feature_columns)
    aligned = aligned.drop(columns=[column for column in ["Final_Label"] if column in aligned.columns], errors="ignore")
    cleaned = aligned.dropna()
    if cleaned.empty:
        raise ValueError("No complete feature row available for inference")
    last_row = cleaned.iloc[-1]
    return last_row, pd.to_datetime(aligned.dropna().index[-1])


def _build_features_for_regime(regime_type: str, data: dict[str, pd.DataFrame], primary_symbol: str) -> pd.DataFrame:
    """Route fetched market data through the shared feature-engineering functions."""

    if regime_type == "trend":
        return feature_engineering.build_trend_features(data[primary_symbol])
    if regime_type == "vol":
        return feature_engineering.build_vol_features(data[primary_symbol], data["^VIX"])
    if regime_type == "bull_bear":
        return feature_engineering.build_bull_bear_features(data[primary_symbol], data["QQQ"], data["IWM"], data["GLD"], data["TLT"], data["^VIX"])
    raise ValueError(f"Unknown regime_type: {regime_type}")


def predict_regime(ticker: str, regime_type: str, registry: ModelRegistry | None = None, as_of_date: str | None = None) -> dict[str, Any]:
    """Predict one regime label for a ticker using the cached production artifacts."""

    start_time = datetime.now()
    registry = registry or ModelRegistry.get_instance()
    if not registry.is_ready():
        raise RuntimeError("Model registry is not ready")

    recent_data = fetch_recent_data(ticker, regime_type=regime_type, as_of_date=as_of_date)
    primary_symbol = ticker if ticker in recent_data else next(iter(recent_data.keys()))
    feature_frame = _build_features_for_regime(regime_type, recent_data, primary_symbol)
    feature_columns = registry.feature_columns.get(regime_type, [column for column in feature_frame.columns if column != "Final_Label"])

    feature_row, inference_date = _prepare_feature_row(feature_frame, feature_columns)
    scaler = registry.scalers[regime_type]
    model = registry.models[regime_type]
    scaled_values = scaler.transform(feature_row.to_frame().T)
    scaled_row = pd.DataFrame(scaled_values, columns=feature_row.index, index=[feature_row.name if feature_row.name is not None else 0])

    prediction = model.predict(scaled_row)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_row)[0]
    else:
        probabilities = np.array([1.0, 0.0])

    latency_ms = (datetime.now() - start_time).total_seconds() * 1000.0
    class_labels = registry.classes.get(regime_type, [])
    if class_labels and isinstance(prediction, (int, np.integer)) and int(prediction) < len(class_labels):
        predicted_label = class_labels[int(prediction)]
    else:
        predicted_label = prediction if isinstance(prediction, str) else str(prediction)
    confidence = float(np.max(probabilities))

    # Interpret confidence level for the frontend and API consumers.
    # During training we dropped 'Neutral' rows where labelling methods disagreed.
    # In production that uncertainty surfaces as low model confidence instead.
    # Thresholds: < 0.60 = uncertain, 0.60-0.75 = moderate, > 0.75 = high confidence.
    if confidence >= 0.75:
        confidence_level = "HIGH"
    elif confidence >= 0.60:
        confidence_level = "MODERATE"
    else:
        confidence_level = "LOW — treat as uncertain"

    return {
        "ticker": ticker,
        "regime_type": regime_type,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "proba_class_0": float(probabilities[0]) if len(probabilities) > 0 else 0.0,
        "proba_class_1": float(probabilities[1]) if len(probabilities) > 1 else confidence,
        "inference_date": str(inference_date.date()),
        "features_used": len(feature_row),
        "inference_latency_ms": float(latency_ms),
        "features_snapshot": feature_row.to_dict(),
    }


def predict_all(ticker: str, as_of_date: str | None = None) -> dict[str, Any]:
    """Predict all three regime labels for a ticker."""

    registry = ModelRegistry.get_instance()
    start = datetime.now()
    results = {
        "trend": predict_regime(ticker, "trend", registry=registry, as_of_date=as_of_date),
        "vol": predict_regime(ticker, "vol", registry=registry, as_of_date=as_of_date),
        "bull_bear": predict_regime(ticker, "bull_bear", registry=registry, as_of_date=as_of_date),
    }
    total_latency = (datetime.now() - start).total_seconds() * 1000.0
    logger.info("Predicted all regimes for %s in %.1f ms", ticker, total_latency)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manual inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol to predict.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    registry = ModelRegistry.get_instance()
    results = predict_all(args.ticker)
    print("=" * 60)
    for regime, payload in results.items():
        print(f"{regime.upper()}: {payload['predicted_label']} | confidence={payload['confidence']:.3f} | latency={payload['inference_latency_ms']:.1f} ms")
    print("=" * 60)