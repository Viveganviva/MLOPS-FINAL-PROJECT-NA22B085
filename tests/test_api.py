from api.schemas import HealthResponse, PredictionRequest


def test_prediction_request_defaults_to_trend_regime():
    request = PredictionRequest(ticker="SPY")
    assert request.regime_type == "trend"


def test_health_response_has_expected_shape():
    response = HealthResponse(status="ok", service="api")
    assert response.status == "ok"