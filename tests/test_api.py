# pytest tests/ -v --tb=short 2>&1 | Tee-Object docs/test_report.txt in powershell for better readability of test results.

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api import main as api_main


class DummyRegistry:
    def __init__(self):
        self.models = {"trend": object(), "vol": object(), "bull_bear": object()}

    def is_ready(self):
        return True


@pytest.fixture()
def client(monkeypatch):
    registry = DummyRegistry()
    monkeypatch.setattr(api_main.monitoring, "start_metrics_server", lambda port=8001: None)
    monkeypatch.setattr(api_main.predict.ModelRegistry, "get_instance", classmethod(lambda cls: registry))
    monkeypatch.setattr(api_main.predict, "predict_regime", lambda ticker, regime_type, registry=None, as_of_date=None: {
        "ticker": ticker,
        "regime_type": regime_type,
        "predicted_label": {"trend": "Trending", "vol": "LowVol", "bull_bear": "Bull"}[regime_type],
        "confidence": 0.91,
        "confidence_level": "HIGH",
        "proba_class_0": 0.09,
        "proba_class_1": 0.91,
        "inference_date": "2024-01-02",
        "features_used": 10,
        "inference_latency_ms": 12.3,
        "features_snapshot": {"feature_1": 1.0},
    })
    monkeypatch.setattr(api_main.drift_monitor, "check_drift_from_log", lambda regime_type: [
        {
            "feature_name": "feature_1",
            "training_mean": 0.1,
            "training_std": 0.2,
            "recent_mean": 0.3,
            "recent_std": 0.4,
            "kl_score": 0.05,
            "drift_detected": False,
        }
    ])
    monkeypatch.setattr(api_main.drift_monitor, "should_retrain", lambda regime_type: (False, "No significant drift detected"))
    return TestClient(api_main.app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data


class TestReadyEndpoint:
    def test_ready_endpoint_exists(self, client):
        response = client.get("/ready")
        assert response.status_code in [200, 503]


class TestPredictEndpoint:
    def test_predict_returns_all_regime_types(self, client):
        response = client.post("/predict", json={"ticker": "SPY", "regime_types": ["trend", "vol", "bull_bear"]})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "trend" in data["results"]
        assert "vol" in data["results"]
        assert "bull_bear" in data["results"]

    def test_invalid_ticker_returns_500(self, client, monkeypatch):
        def failing_predict(*_args, **_kwargs):
            raise ValueError("invalid ticker")

        monkeypatch.setattr(api_main.predict, "predict_regime", failing_predict)
        response = client.post("/predict", json={"ticker": "INVALIDTICKER999", "regime_types": ["trend"]})
        assert response.status_code == 500
