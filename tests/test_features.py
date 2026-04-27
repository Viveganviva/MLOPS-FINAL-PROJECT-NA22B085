from src.feature_engineering import build_features


def test_build_features_accepts_parameter_mapping():
    build_features({"data": {"processed_dir": "data/processed"}, "features": {"short_window": 5, "mid_window": 20, "long_window": 50}})