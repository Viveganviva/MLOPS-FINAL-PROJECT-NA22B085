from src.predict import predict_regime


def test_predict_regime_returns_placeholder_prediction():
    result = predict_regime({"close": 1.0, "volume": 2.0})
    assert result["predicted_label"] == "unknown"