from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import build_trend_features, compute_rsi


def make_market_frame(periods: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=periods)
    close = pd.Series(np.linspace(100, 180, periods), index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": np.linspace(1_000_000, 1_250_000, periods),
        },
        index=index,
    )


class TestRSIComputation:
    def test_rsi_bounded_0_100(self):
        series = pd.Series(np.linspace(100, 140, 80))
        rsi = compute_rsi(series, period=14).dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_rsi_overbought_on_rising_prices(self):
        series = pd.Series(np.linspace(100, 200, 120))
        rsi = compute_rsi(series, period=14).dropna()
        assert rsi.iloc[-1] > 70


class TestFeatureBuilding:
    def test_no_label_leakage_in_trend_features(self):
        trend = build_trend_features(make_market_frame())
        cols = trend.columns
        assert "Label_Hurst" not in cols
        assert "Label_VR" not in cols
        assert "Label_Autocorr" not in cols

    def test_feature_count_matches_expectation(self):
        trend = build_trend_features(make_market_frame())
        assert len([column for column in trend.columns if column != "Final_Label"]) >= 15

    def test_no_nan_in_output(self):
        trend = build_trend_features(make_market_frame())
        assert not trend.isna().any().any()
