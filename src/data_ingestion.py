"""Download and validate market data for the regime detection pipeline.

The module keeps all operational configuration in params.yaml and is designed
to be safe to run repeatedly during development. Training data is skipped when
the cached files are still fresh enough, while simulation data is always kept
separate from training inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency guard for clarity during bootstrap
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - handled at runtime with context
    yf = None  # type: ignore[assignment]
    _YFINANCE_IMPORT_ERROR = exc
else:
    _YFINANCE_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PREDICTION_LOG_HEADERS = [
    "timestamp",
    "ticker",
    "regime_type",
    "predicted_label",
    "confidence",
    "features_snapshot",
]


def _load_params() -> dict[str, Any]:
    """Load the project parameter map from params.yaml at module import time."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Parameter file not found: {PARAMS_PATH}")

    with PARAMS_PATH.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}

    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")

    return params


PARAMS = _load_params()
DATA_PARAMS = PARAMS["data"]
INGESTION_PARAMS = PARAMS["ingestion"]
RAW_DIR = PROJECT_ROOT / DATA_PARAMS["raw_dir"]
BASELINE_DIR = PROJECT_ROOT / DATA_PARAMS["baseline_dir"]
SIM_DIR = PROJECT_ROOT / DATA_PARAMS["sim_dir"]
TRAIN_TICKERS = list(DATA_PARAMS["train_tickers"])
PRIMARY_TICKER = DATA_PARAMS["primary_ticker"]
VIX_TICKER = DATA_PARAMS["vix_ticker"]
TRAIN_START = DATA_PARAMS["train_start"]
TRAIN_END = DATA_PARAMS["train_end"]
SIM_START = DATA_PARAMS["sim_start"]
SIM_END = DATA_PARAMS["sim_end"]
DOWNLOAD_RETRIES = int(INGESTION_PARAMS["download_retries"])
RETRY_SLEEP_SECONDS = int(INGESTION_PARAMS["retry_sleep_seconds"])
FRESHNESS_WINDOW = timedelta(hours=int(INGESTION_PARAMS["freshness_hours"]))
VALIDATION_TOLERANCE_DAYS = int(INGESTION_PARAMS["validation_tolerance_days"])
LOG_FORMAT = INGESTION_PARAMS["log_format"]
LOG_DATEFMT = INGESTION_PARAMS["log_datefmt"]


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console and file logging for ingestion runs.

    The log directory is created eagerly so the first run succeeds even when the
    repository starts from a clean checkout.
    """

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "ingestion.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def _ensure_yfinance_available() -> None:
    """Raise a clear runtime error if yfinance is unavailable."""

    if yf is None:  # pragma: no cover - environment dependent
        raise RuntimeError("yfinance is required for data ingestion.") from _YFINANCE_IMPORT_ERROR


def _parse_timestamp(value: Any) -> pd.Timestamp:
    """Convert a date-like value into a normalized pandas timestamp."""

    return pd.to_datetime(value, utc=False).tz_localize(None)


def _format_missing_pct(frame: pd.DataFrame) -> float:
    """Compute the average missing-value percentage across columns."""

    if frame.empty:
        return 100.0

    return float(frame.isna().mean().mean() * 100.0)


def _build_validation(ticker: str, frame: pd.DataFrame, expected_start: str, expected_end: str) -> dict[str, Any]:
    """Summarize the health of a downloaded ticker frame."""

    actual_start = frame.index.min() if not frame.empty else None
    actual_end = frame.index.max() if not frame.empty else None
    missing_pct = _format_missing_pct(frame)

    status = "PASS"
    problems: list[str] = []

    if frame.empty:
        status = "FAIL"
        problems.append("empty frame")

    all_nan_columns = [column for column in frame.columns if frame[column].isna().all()]
    if all_nan_columns:
        status = "FAIL"
        problems.append(f"all-NaN columns: {', '.join(all_nan_columns)}")

    expected_start_ts = _parse_timestamp(expected_start)
    expected_end_ts = _parse_timestamp(expected_end)
    if actual_start is not None and actual_start > expected_start_ts + pd.Timedelta(days=VALIDATION_TOLERANCE_DAYS):
        status = "FAIL"
        problems.append("start date too far from request")
    if actual_end is not None and actual_end < expected_end_ts - pd.Timedelta(days=VALIDATION_TOLERANCE_DAYS):
        status = "FAIL"
        problems.append("end date too far from request")

    if problems:
        logger.warning("Validation issues for %s: %s", ticker, "; ".join(problems))

    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "date_start": actual_start.date().isoformat() if actual_start is not None else "NA",
        "date_end": actual_end.date().isoformat() if actual_end is not None else "NA",
        "missing_pct": round(missing_pct, 2),
        "status": status,
    }


def validate_data(frame: pd.DataFrame, ticker: str, expected_start: str | None = None, expected_end: str | None = None) -> dict[str, Any]:
    """Public wrapper for frame validation used by tests and downstream tooling."""

    if expected_start is None and not frame.empty:
        expected_start = pd.to_datetime(frame.index.min()).date().isoformat()
    if expected_end is None and not frame.empty:
        expected_end = pd.to_datetime(frame.index.max()).date().isoformat()
    expected_start = expected_start or TRAIN_START
    expected_end = expected_end or TRAIN_END
    return _build_validation(ticker, frame, expected_start, expected_end)


def _save_frame(frame: pd.DataFrame, save_path: Path) -> None:
    """Persist a downloaded frame to disk with the date index preserved."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = frame.copy()
    to_write.index.name = "date"
    to_write.to_csv(save_path, index=True)


def _is_fresh(path: Path) -> tuple[bool, float | None]:
    """Return whether a cached file is fresh enough to skip downloading."""

    if not path.exists():
        return False, None

    age_seconds = datetime.now(timezone.utc).timestamp() - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).timestamp()
    age_hours = age_seconds / 3600.0
    return age_hours <= FRESHNESS_WINDOW.total_seconds() / 3600.0, age_hours


def _read_cached_frame(path: Path) -> pd.DataFrame:
    """Load an existing CSV using the date column as the index."""

    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame


def download_ticker(ticker: str, start: str, end: str, save_path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download a ticker's OHLCV history, validate it, and persist it to CSV.

    The download is retried because Yahoo Finance requests can fail transiently.
    The returned validation dictionary is used by the training and simulation
    reporting flows.
    """

    _ensure_yfinance_available()

    destination = Path(save_path)
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            frame = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, actions=False)
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            # Also drop any duplicate columns that can appear after flattening
            frame = frame.loc[:, ~frame.columns.duplicated()]

            if frame.empty:
                raise ValueError(f"downloaded frame for {ticker} is empty")

            frame = frame.dropna(axis=1, how="all")
            if frame.empty:
                raise ValueError(f"downloaded frame for {ticker} contains only NaN columns")

            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            validation = _build_validation(ticker, frame, start, end)
            if validation["status"] != "PASS":
                raise ValueError(f"validation failed for {ticker}: {validation}")

            _save_frame(frame, destination)
            logger.info("Downloaded %s to %s", ticker, destination)
            return frame, validation
        except Exception as exc:
            last_error = exc
            logger.exception("Attempt %s/%s failed for %s: %s", attempt, DOWNLOAD_RETRIES, ticker, exc)
            if attempt < DOWNLOAD_RETRIES:
                sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Failed to download {ticker} after {DOWNLOAD_RETRIES} attempts") from last_error


def _print_validation_table(validations: list[dict[str, Any]]) -> None:
    """Render a compact validation table for the completed downloads."""

    headers = ["Ticker", "Rows", "Start", "End", "Missing %", "Status"]
    rows = [
        [
            item["ticker"],
            str(item["rows"]),
            item["date_start"],
            item["date_end"],
            f'{item["missing_pct"]:.2f}%',
            f'  {item["status"]}  ',
        ]
        for item in validations
    ]

    widths = [
        max(len(headers[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(len(headers))
    ]

    def border(left: str, middle: str, right: str) -> str:
        segments = ["═" * (width + 2) for width in widths]
        return left + middle.join(segments) + right

    def render_row(values: list[str]) -> str:
        cells = [f" {value.ljust(widths[index])} " for index, value in enumerate(values)]
        return "║" + "║".join(cells) + "║"

    print(border("╔", "╦", "╗"))
    print(render_row(headers))
    print(border("╠", "╬", "╣"))
    for row in rows:
        print(render_row(row))
    print(border("╚", "╩", "╝"))


def download_training_data() -> dict[str, Any]:
    """Download the training window for all configured tickers and the VIX.

    Existing files younger than 24 hours are reused to avoid unnecessary
    re-downloads during active development sessions.
    """

    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    validations: list[dict[str, Any]] = []
    fresh_files = 0

    for ticker in TRAIN_TICKERS:
        save_path = raw_dir / f"{ticker}_train.csv"
        fresh, age_hours = _is_fresh(save_path)
        if fresh:
            age_text = f"{age_hours:.1f} hours ago" if age_hours is not None else "recently"
            print(f"[INFO] {ticker}_train.csv already fresh (downloaded {age_text}). Skipping.")
            frame = _read_cached_frame(save_path)
            validations.append(_build_validation(ticker, frame, TRAIN_START, TRAIN_END))
            fresh_files += 1
            continue

        frame, validation = download_ticker(ticker, TRAIN_START, TRAIN_END, save_path)
        validations.append(validation)

    vix_path = raw_dir / "VIX_train.csv"
    fresh, age_hours = _is_fresh(vix_path)
    if fresh:
        age_text = f"{age_hours:.1f} hours ago" if age_hours is not None else "recently"
        print(f"[INFO] VIX_train.csv already fresh (downloaded {age_text}). Skipping.")
        frame = _read_cached_frame(vix_path)
        validations.append(_build_validation("VIX", frame, TRAIN_START, TRAIN_END))
        fresh_files += 1
    else:
        frame, validation = download_ticker(VIX_TICKER, TRAIN_START, TRAIN_END, vix_path)
        validations.append(validation)

    _print_validation_table(validations)
    logger.info("Training data validation completed for %d files", len(validations))

    return {
        "validations": validations,
        "fresh_files": fresh_files,
        "training_rows": {item["ticker"]: item["rows"] for item in validations},
    }


def download_simulation_data() -> dict[str, Any]:
    """Download the held-out simulation window for SPY and VIX only."""

    SIM_DIR.mkdir(parents=True, exist_ok=True)

    spy_path = SIM_DIR / "SPY_sim.csv"
    vix_path = SIM_DIR / "VIX_sim.csv"

    spy_frame, _ = download_ticker(PRIMARY_TICKER, SIM_START, SIM_END, spy_path)
    vix_frame, _ = download_ticker(VIX_TICKER, SIM_START, SIM_END, vix_path)

    print(f"[SIMULATION] Downloaded {len(spy_frame)} trading days for live simulation (2023–2024)")
    logger.info("Simulation data saved to %s and %s", spy_path, vix_path)

    return {
        "spy_rows": int(len(spy_frame)),
        "vix_rows": int(len(vix_frame)),
        "total_rows": int(len(spy_frame)),
    }


def compute_baseline_stats() -> dict[str, Any]:
    """Compute baseline distribution statistics from the training SPY dataset."""

    spy_path = RAW_DIR / "SPY_train.csv"
    if not spy_path.exists():
        raise FileNotFoundError(f"Training file not found: {spy_path}")

    spy_frame = _read_cached_frame(spy_path)
    close_prices = pd.to_numeric(spy_frame["Close"], errors="coerce")
    volume = pd.to_numeric(spy_frame["Volume"], errors="coerce")
    daily_log_return = np.log(close_prices / close_prices.shift(1))
    rolling_std = daily_log_return.rolling(window=20, min_periods=1).std()

    baseline_stats: dict[str, dict[str, float]] = {}
    for name, series in {
        "Close": close_prices,
        "Volume": volume,
        "daily_log_return": daily_log_return,
        "20_day_rolling_std": rolling_std,
    }.items():
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        baseline_stats[name] = {
            "mean": float(numeric_series.mean()),
            "std": float(numeric_series.std(ddof=1)),
            "min": float(numeric_series.min()),
            "25%": float(numeric_series.quantile(0.25)),
            "50%": float(numeric_series.quantile(0.50)),
            "75%": float(numeric_series.quantile(0.75)),
            "max": float(numeric_series.max()),
        }

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = BASELINE_DIR / "feature_baselines.json"
    with baseline_path.open("w", encoding="utf-8") as handle:
        json.dump(baseline_stats, handle, indent=2, sort_keys=True)

    print("[BASELINE] Saved feature distribution stats to data/baselines/feature_baselines.json")
    logger.info("Baseline statistics written to %s", baseline_path)

    return {"path": baseline_path, "stats": baseline_stats}


def _count_successful_tickers(validations: list[dict[str, Any]]) -> tuple[int, int]:
    """Count the number of successful ticker validations."""

    success_count = sum(1 for item in validations if item["status"] == "PASS")
    return success_count, len(validations)


def main() -> None:
    """Run the full ingestion workflow and print a concise completion summary."""

    setup_logging()

    started = datetime.now(timezone.utc)
    print("=" * 48)
    print(" MARKET REGIME DETECTION — DATA INGESTION")
    print(f" Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 48)

    try:
        training_result = download_training_data()
        simulation_result = download_simulation_data()
        baseline_result = compute_baseline_stats()

        spy_train_path = RAW_DIR / "SPY_train.csv"
        spy_rows = 0
        if spy_train_path.exists():
            spy_rows = int(len(_read_cached_frame(spy_train_path)))

        successful_tickers, total_tickers = _count_successful_tickers(training_result["validations"])

        print("=" * 48)
        print(" INGESTION COMPLETE")
        print(f" Training rows (SPY): {spy_rows}")
        print(f" Simulation rows:     {simulation_result['total_rows']}")
        print(f" Tickers successful:  {successful_tickers}/{total_tickers}")
        print(f" Baseline stats saved: {'YES' if baseline_result['path'].exists() else 'NO'}")
        print(" Next step: python src/feature_engineering.py")
        print("=" * 48)
    except Exception as exc:
        logger.exception("Data ingestion failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()