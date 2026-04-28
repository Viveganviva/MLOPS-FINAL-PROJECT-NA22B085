from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import predict


class DummyScaler:
    def transform(self, frame):
        return frame.to_numpy(dtype=float)


class DummyModel:
    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.22, 0.78]])


class DummyRegistry:
    def __init__(self):
        self.models = {"trend": DummyModel(), "vol": DummyModel(), "bull_bear": DummyModel()}
        self.scalers = {"trend": DummyScaler(), "vol": DummyScaler(), "bull_bear": DummyScaler()}
        self.feature_columns = {
            "trend": ["feature_1", "feature_2"],
            "vol": ["feature_1", "feature_2"],
            "bull_bear": ["feature_1", "feature_2"],
        }
        self.classes = {
            "trend": ["Trending", "MeanReverting"],
            "vol": ["HighVol", "LowVol"],
            "bull_bear": ["Bull", "Bear"],
        }

    def is_ready(self):
        return True


def make_feature_frame():
    return pd.DataFrame({"feature_1": [1.0], "feature_2": [2.0], "Final_Label": ["Trending"]}, index=[pd.Timestamp("2024-01-02")])


class TestPredictionOutput:
    def test_prediction_returns_expected_keys(self, monkeypatch):
        registry = DummyRegistry()
        monkeypatch.setattr(predict, "fetch_recent_data", lambda *args, **kwargs: {"SPY": pd.DataFrame()})
        monkeypatch.setattr(predict, "_build_features_for_regime", lambda *args, **kwargs: make_feature_frame())
        monkeypatch.setattr(predict.ModelRegistry, "get_instance", classmethod(lambda cls: registry))

        result = predict.predict_regime("SPY", "trend")

        assert {"ticker", "regime_type", "predicted_label", "confidence", "features_snapshot"}.issubset(result.keys())

    def test_confidence_between_0_and_1(self, monkeypatch):
        registry = DummyRegistry()
        monkeypatch.setattr(predict, "fetch_recent_data", lambda *args, **kwargs: {"SPY": pd.DataFrame()})
        monkeypatch.setattr(predict, "_build_features_for_regime", lambda *args, **kwargs: make_feature_frame())
        monkeypatch.setattr(predict.ModelRegistry, "get_instance", classmethod(lambda cls: registry))

        result = predict.predict_regime("SPY", "vol")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predicted_label_is_valid_class(self, monkeypatch):
        registry = DummyRegistry()
        monkeypatch.setattr(predict, "fetch_recent_data", lambda *args, **kwargs: {"SPY": pd.DataFrame()})
        monkeypatch.setattr(predict, "_build_features_for_regime", lambda *args, **kwargs: make_feature_frame())
        monkeypatch.setattr(predict.ModelRegistry, "get_instance", classmethod(lambda cls: registry))

        valid_labels = {
            "trend": ["Trending", "MeanReverting"],
            "vol": ["HighVol", "LowVol"],
            "bull_bear": ["Bull", "Bear"],
        }

        for regime, labels in valid_labels.items():
            result = predict.predict_regime("SPY", regime)
            assert result["predicted_label"] in labels
