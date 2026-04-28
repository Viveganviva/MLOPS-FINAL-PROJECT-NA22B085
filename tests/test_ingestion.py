from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_ingestion


class DummyYFinance:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def download(self, *_args, **_kwargs):
        return self.frame.copy()


def make_ohlcv_frame(start: str = "2023-01-01", periods: int = 25) -> pd.DataFrame:
    index = pd.bdate_range(start=start, periods=periods)
    close = pd.Series(np.linspace(100, 120, periods), index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": np.linspace(1_000_000, 1_200_000, periods),
        },
        index=index,
    )


class TestDownloadTicker:
    def test_returns_dataframe_with_expected_columns(self, tmp_path, monkeypatch):
        frame = make_ohlcv_frame()
        monkeypatch.setattr(data_ingestion, "yf", DummyYFinance(frame))
        monkeypatch.setattr(data_ingestion, "DOWNLOAD_RETRIES", 1)
        output_path = tmp_path / "spy.csv"

        downloaded, validation = data_ingestion.download_ticker("SPY", "2023-01-01", "2023-01-31", output_path)

        assert isinstance(downloaded, pd.DataFrame)
        assert all(column in downloaded.columns for column in ["Open", "High", "Low", "Close", "Volume"])
        assert validation["status"] == "PASS"
        assert output_path.exists()

    def test_validation_catches_empty_dataframe(self):
        result = data_ingestion.validate_data(pd.DataFrame(), "TEST")
        assert result["status"] == "FAIL"

    def test_date_range_coverage(self, tmp_path, monkeypatch):
        frame = make_ohlcv_frame(periods=22)
        monkeypatch.setattr(data_ingestion, "yf", DummyYFinance(frame))
        monkeypatch.setattr(data_ingestion, "DOWNLOAD_RETRIES", 1)
        downloaded, _ = data_ingestion.download_ticker("SPY", "2023-01-01", "2023-01-31", tmp_path / "spy.csv")

        requested = pd.bdate_range("2023-01-01", "2023-01-31")
        coverage = len(downloaded) / len(requested)
        assert coverage >= 0.9


class TestDataQuality:
    def test_no_negative_prices(self):
        frame = make_ohlcv_frame()
        result = data_ingestion.validate_data(frame, "TEST")
        assert result["status"] == "PASS"
        assert (frame[["Open", "High", "Low", "Close"]] > 0).all().all()

    def test_baseline_stats_file_created(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        baseline_dir = tmp_path / "baselines"
        raw_dir.mkdir(parents=True)
        baseline_dir.mkdir(parents=True)
        make_ohlcv_frame().to_csv(raw_dir / "SPY_train.csv")

        monkeypatch.setattr(data_ingestion, "RAW_DIR", raw_dir)
        monkeypatch.setattr(data_ingestion, "BASELINE_DIR", baseline_dir)

        result = data_ingestion.compute_baseline_stats()
        assert result["path"].exists()
        with result["path"].open("r", encoding="utf-8") as handle:
            baseline_json = json.load(handle)
        assert "Close" in baseline_json
