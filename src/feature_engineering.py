"""Feature engineering and regime labeling for the market regime pipeline.

This module is the single source of truth for feature computation. The same
functions are used for training and must remain importable for live inference in
the API layer.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency guard for clean imports
    from arch import arch_model
except ImportError as exc:  # pragma: no cover - surfaced when GARCH is used
    arch_model = None  # type: ignore[assignment]
    _ARCH_IMPORT_ERROR = exc
else:
    _ARCH_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

def _ensure_flat_ohlcv(frame):
    """
    Defensive normaliser called at the start of every function that accepts
    an OHLCV DataFrame. yfinance >= 0.2.18 returns MultiIndex columns like
    ('Close', 'SPY') even for single-ticker downloads. This collapses any
    such index down to a flat single level so frame['Close'] always works.
    Safe to call multiple times.
    """
    import pandas as _pd
    if frame is None:
        return frame
    if not hasattr(frame, "columns"):
        return frame
    if isinstance(frame.columns, _pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
        frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame


def _load_params() -> dict[str, Any]:
    """Load the project parameter map from params.yaml at import time."""

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
FEATURE_PARAMS = PARAMS["features"]
RAW_DIR = PROJECT_ROOT / DATA_PARAMS["raw_dir"]
PROCESSED_DIR = PROJECT_ROOT / DATA_PARAMS["processed_dir"]
BASELINE_DIR = PROJECT_ROOT / DATA_PARAMS["baseline_dir"]
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

RSI_PERIOD = int(FEATURE_PARAMS["rsi_period"])
ATR_PERIOD = int(FEATURE_PARAMS["atr_period"])
MACD_FAST = int(FEATURE_PARAMS["macd_fast"])
MACD_SLOW = int(FEATURE_PARAMS["macd_slow"])
MACD_SIGNAL = int(FEATURE_PARAMS["macd_signal"])
HURST_WINDOW = int(FEATURE_PARAMS["hurst_window"])
VR_WINDOW = int(FEATURE_PARAMS["vr_window"])
VR_K = int(FEATURE_PARAMS["vr_k"])
AUTOCORR_WINDOW = int(FEATURE_PARAMS["autocorr_window"])
SHORT_WINDOW = int(FEATURE_PARAMS["short_window"])
MID_WINDOW = int(FEATURE_PARAMS["mid_window"])
LONG_WINDOW = int(FEATURE_PARAMS["long_window"])

logger = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance >= 0.2.18 returns MultiIndex columns like ('Close', 'SPY') even
    for single-ticker downloads. Flatten them to simple strings like 'Close'.
    Safe to call even if columns are already flat.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns that yfinance >= 0.2.18 returns even for
    single-ticker downloads.  After download('SPY', ...) the columns look
    like [('Close','SPY'), ('High','SPY'), ...].  We drop the ticker level
    so downstream code can always access df['Close'] etc.
    Also deduplicates any columns that appear twice after flattening.
    """
    df = _flatten_columns(df)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console and file logging for feature generation runs."""

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "features.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    """Load a params.yaml file for ad hoc usage or tests."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")

    return config


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with a normalized datetime index and numeric OHLCV columns."""
    df = _flatten_columns(df)
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()

    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _load_csv_frame(path: Path) -> pd.DataFrame:
    """Load a raw CSV written by the ingestion stage."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    return _prepare_frame(frame)


def _load_raw_frames() -> dict[str, pd.DataFrame]:
    """Load all market data files required by the training feature builders."""

    return {
        "SPY": _load_csv_frame(RAW_DIR / "SPY_train.csv"),
        "QQQ": _load_csv_frame(RAW_DIR / "QQQ_train.csv"),
        "IWM": _load_csv_frame(RAW_DIR / "IWM_train.csv"),
        "GLD": _load_csv_frame(RAW_DIR / "GLD_train.csv"),
        "TLT": _load_csv_frame(RAW_DIR / "TLT_train.csv"),
        "VIX": _load_csv_frame(RAW_DIR / "VIX_train.csv"),
    }


def _ensure_arch_available() -> None:
    """Log a warning when the arch package is not installed."""

    if arch_model is None:  # pragma: no cover - environment dependent
        logger.warning(
            "arch is not installed in the active environment; GARCH labels will use a realized-volatility fallback"
        )


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI using Wilder's smoothing rather than a simple moving average."""

    # Using manual implementation for full transparency and to avoid dependency on TA-Lib which requires C compiler setup
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi")


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Compute the Average True Range using Wilder-style smoothing."""

    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr.rename("atr")


def compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Compute the MACD line, signal line, and histogram."""

    close = pd.to_numeric(close, errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }
    )


def compute_bollinger_bands(close: pd.Series, window: int, num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands and the associated normalized bandwidth measures."""

    close = pd.to_numeric(close, errors="coerce")
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower)

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_middle": middle,
            "bb_width": width,
            "bb_pct_b": pct_b,
        }
    )


def _hurst_rs(values: np.ndarray) -> float:
    """Estimate the Hurst exponent using a simple rescaled-range statistic."""

    series = pd.Series(values).dropna()
    if len(series) < 3:
        return np.nan

    deviations = series - series.mean()
    cumulative = deviations.cumsum()
    range_stat = cumulative.max() - cumulative.min()
    scale = series.std(ddof=1)
    if not np.isfinite(range_stat) or not np.isfinite(scale) or scale <= 0 or range_stat <= 0:
        return np.nan

    return float(np.log(range_stat / scale) / np.log(len(series)))


def compute_hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling Hurst exponents for regime labeling only."""

    # Used ONLY for label generation, not as a feature — using it as a feature would cause data leakage since Hurst directly encodes the regime label
    close = pd.to_numeric(series, errors="coerce")
    return close.rolling(window=window, min_periods=window).apply(_hurst_rs, raw=True).rename("hurst")


def _variance_ratio_window(values: np.ndarray, k: int) -> float:
    """Compute a rolling variance ratio over a price window."""

    prices = pd.Series(values).dropna().astype(float)
    if len(prices) <= k + 1:
        return np.nan

    log_prices = np.log(prices)
    one_period_returns = log_prices.diff().dropna()
    if len(one_period_returns) < 2 or np.isclose(one_period_returns.var(ddof=1), 0.0):
        return np.nan

    k_period_returns = log_prices.diff(k).dropna()
    var_one = one_period_returns.var(ddof=1)
    var_k = k_period_returns.var(ddof=1)
    if not np.isfinite(var_one) or var_one <= 0 or not np.isfinite(var_k):
        return np.nan

    return float(var_k / (k * var_one))


def compute_variance_ratio(series: pd.Series, window: int, k: int) -> pd.Series:
    """Compute rolling variance ratio values for regime labeling only."""

    # Used ONLY for label generation, not as a feature
    close = pd.to_numeric(series, errors="coerce")
    return close.rolling(window=window, min_periods=window).apply(_variance_ratio_window, raw=True, args=(k,)).rename("variance_ratio")


def _rolling_autocorr(series: pd.Series, window: int) -> pd.Series:
    """Compute the one-lag autocorrelation over a rolling window."""

    def _autocorr(values: np.ndarray) -> float:
        window_series = pd.Series(values).dropna()
        if len(window_series) < 3:
            return np.nan
        return float(window_series.autocorr(lag=1))

    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=window).apply(_autocorr, raw=True)


def _trend_label_from_signals(hurst: float, variance_ratio: float, autocorr: float) -> str:
    """Combine regime signals into a single trend label with majority vote."""

    votes = []
    if pd.notna(hurst):
        votes.append("Trending" if hurst >= 0.55 else "MeanReverting" if hurst <= 0.45 else "Neutral")
    if pd.notna(variance_ratio):
        votes.append("Trending" if variance_ratio >= 1.05 else "MeanReverting" if variance_ratio <= 0.95 else "Neutral")
    if pd.notna(autocorr):
        votes.append("Trending" if autocorr >= 0.10 else "MeanReverting" if autocorr <= -0.10 else "Neutral")

    if not votes:
        return "Neutral"

    counts = Counter(votes)
    best_label, best_count = counts.most_common(1)[0]
    tied = [label for label, count in counts.items() if count == best_count]
    if len(tied) > 1:
        return "Neutral"
    return best_label


def label_trend_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Label trend versus mean-reversion regimes using only label-safe signals."""
    
    df = _flatten_columns(df)
    frame = _normalise_ohlcv(df)
    frame = _ensure_flat_ohlcv(frame)
    frame = _prepare_frame(frame)
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)
        frame = frame.loc[:, ~frame.columns.duplicated()]
    close = frame["Close"]

    close = frame["Close"]

    frame["Label_Hurst"] = compute_hurst_exponent(close, HURST_WINDOW)
    frame["Label_VR"] = compute_variance_ratio(close, VR_WINDOW, VR_K)
    log_returns = np.log(close / close.shift(1))
    frame["Label_Autocorr"] = _rolling_autocorr(log_returns, AUTOCORR_WINDOW)

    frame["Final_Label"] = frame.apply(
        lambda row: _trend_label_from_signals(row["Label_Hurst"], row["Label_VR"], row["Label_Autocorr"]),
        axis=1,
    )

    label_distribution = frame["Final_Label"].value_counts(dropna=False).to_dict()
    logger.info("Trend regime label distribution: %s", label_distribution)

    return frame.drop(columns=["Label_Hurst", "Label_VR", "Label_Autocorr"])


def _fit_garch_conditional_volatility(returns: pd.Series) -> pd.Series:
    """Fit a GARCH(1,1) model and return the conditional volatility series."""

    if arch_model is None:  # pragma: no cover - environment dependent
        _ensure_arch_available()
        return (returns.rolling(window=20, min_periods=20).std() * np.sqrt(252.0)).rename("garch_conditional_vol")

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna() * 100.0
    if len(clean_returns) < 100:
        return (returns.rolling(window=20, min_periods=20).std() * np.sqrt(252.0)).rename("garch_conditional_vol")

    try:
        model = arch_model(clean_returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        result = model.fit(disp="off")
        conditional_vol = result.conditional_volatility.reindex(clean_returns.index)
        return (conditional_vol / 100.0).rename("garch_conditional_vol")
    except Exception as exc:
        logger.exception("GARCH fit failed, falling back to rolling volatility: %s", exc)
        fallback = returns.rolling(window=20, min_periods=20).std()
        return fallback.rename("garch_conditional_vol")


def _vol_regime_label_from_votes(method_a: bool, method_b: bool, method_c: bool) -> str:
    """Convert the three volatility methods into a single majority label."""

    votes = ["HighVol" if flag else "LowVol" for flag in [method_a, method_b, method_c]]
    counts = Counter(votes)
    high = counts.get("HighVol", 0)
    low = counts.get("LowVol", 0)
    return "HighVol" if high > low else "LowVol"


def label_vol_regime(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Label high- and low-volatility regimes using three independent methods."""
    df = _flatten_columns(df)
    vix_df = _flatten_columns(vix_df)
    frame = _normalise_ohlcv(df)
    frame = _ensure_flat_ohlcv(frame) 
    vix_df = _ensure_flat_ohlcv(vix_df)
    vix_df = _normalise_ohlcv(vix_df)
    spy = _prepare_frame(frame)
    vix = _prepare_frame(vix_df)
    frame = spy.join(vix[["Close"]].rename(columns={"Close": "VIX_Close"}), how="inner")

    returns = np.log(frame["Close"] / frame["Close"].shift(1))
    rolling_std_20 = returns.rolling(window=20, min_periods=20).std()
    rolling_median_252 = rolling_std_20.rolling(window=252, min_periods=126).median()
    vix_high = frame["VIX_Close"] > 20
    garch_vol = _fit_garch_conditional_volatility(returns)
    garch_median = garch_vol.rolling(window=252, min_periods=126).median()

    frame["Label_Method_A"] = rolling_std_20 > rolling_median_252
    frame["Label_Method_B"] = vix_high
    frame["Label_Method_C"] = garch_vol > garch_median
    frame["Final_Label"] = frame.apply(
        lambda row: _vol_regime_label_from_votes(bool(row["Label_Method_A"]), bool(row["Label_Method_B"]), bool(row["Label_Method_C"])),
        axis=1,
    )

    label_distribution = frame["Final_Label"].value_counts(dropna=False).to_dict()
    logger.info("Volatility regime label distribution: %s", label_distribution)

    return frame.drop(columns=["Label_Method_A", "Label_Method_B", "Label_Method_C"])


def label_bull_bear_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Label bull and bear regimes using a rolling peak-to-trough drawdown rule."""
    df = _flatten_columns(df)
    frame = _normalise_ohlcv(df)
    frame = _ensure_flat_ohlcv(frame)
    frame = _prepare_frame(frame)
    rolling_high = frame["Close"].rolling(window=252, min_periods=126).max()
    frame["Final_Label"] = np.where(frame["Close"] < 0.8 * rolling_high, "Bear", "Bull")

    label_distribution = frame["Final_Label"].value_counts(dropna=False).to_dict()
    logger.info("Bull/bear regime label distribution: %s", label_distribution)
    return frame


def _finalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete rows and keep the final label attached to the features."""
    frame = _flatten_columns(frame)
    cleaned = frame.dropna().copy()
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    return cleaned


def build_trend_features(spy_df: pd.DataFrame) -> pd.DataFrame:
    """Build trend-regime features from SPY OHLCV data."""
    spy_df = _ensure_flat_ohlcv(spy_df)
    spy_df = _flatten_columns(spy_df)
    spy_df = _normalise_ohlcv(spy_df)

    # The label uses Hurst/VR/autocorr, so these values are intentionally excluded from the feature set to avoid leakage.
    frame = label_trend_regime(spy_df)

    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"]

    # Momentum and volatility features help separate persistent trends from choppy mean-reverting tape.
    frame["log_return"] = np.log(close / close.shift(1))
    frame["rolling_mean_5"] = close.rolling(window=5).mean()
    frame["rolling_mean_10"] = close.rolling(window=10).mean()
    frame["rolling_mean_20"] = close.rolling(window=20).mean()
    frame["rolling_std_5"] = frame["log_return"].rolling(window=5).std()
    frame["rolling_std_20"] = frame["log_return"].rolling(window=20).std()
    frame["cumret_10"] = close / close.shift(10) - 1
    frame["cumret_20"] = close / close.shift(20) - 1
    frame["rsi"] = compute_rsi(close, RSI_PERIOD)
    macd = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    frame = frame.join(macd)
    bollinger = compute_bollinger_bands(close, window=20, num_std=2.0)
    frame = frame.join(bollinger[["bb_width", "bb_pct_b"]])
    frame["atr"] = compute_atr(high, low, close, ATR_PERIOD)
    frame["roc_10"] = close / close.shift(10) - 1
    frame["momentum_5"] = close - close.shift(5)
    frame["rolling_skew_20"] = frame["log_return"].rolling(window=20).skew()
    frame["rolling_kurt_20"] = frame["log_return"].rolling(window=20).kurt()
    # SMA-50/SMA-200 is safe here because the label is derived from Hurst/VR/autocorrelation, not moving-average structure.
    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()
    frame["sma_50_200_ratio"] = sma_50 / sma_200
    frame["high_low_range"] = (high - low) / close
    frame["volume_change_5"] = volume / volume.rolling(window=5).mean()

    feature_frame = _finalize_feature_frame(frame)
    feature_columns = [column for column in feature_frame.columns if column != "Final_Label"]
    logger.info("Built trend feature frame with %d rows and %d features", len(feature_frame), len(feature_columns))
    return feature_frame


def build_vol_features(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build volatility-regime features from SPY and VIX market data."""
    spy_df = _ensure_flat_ohlcv(spy_df)
    vix_df = _ensure_flat_ohlcv(vix_df)
    spy_df = _flatten_columns(spy_df)
    vix_df = _flatten_columns(vix_df)
    spy_df = _normalise_ohlcv(spy_df)
    vix_df = _normalise_ohlcv(vix_df)

    frame = label_vol_regime(spy_df, vix_df)

    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    vix_close = frame["VIX_Close"]

    # Close-to-close returns capture realized drift, while rolling vol isolates the changing amplitude of shocks.
    frame["log_return"] = np.log(close / close.shift(1))
    frame["rolling_std_5"] = frame["log_return"].rolling(window=5).std()
    frame["rolling_std_20"] = frame["log_return"].rolling(window=20).std()
    # Parkinson volatility uses the intraday high-low range, which is more efficient than close-to-close volatility when range data is available.
    frame["parkinson_vol"] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(high / low) ** 2))
    # Garman-Klass blends range and open-close information to reduce noise relative to a pure close-based estimator.
    frame["garman_klass_vol"] = np.sqrt(
        0.5 * (np.log(high / low) ** 2) - (2 * np.log(2) - 1) * (np.log(close / frame["Open"]) ** 2)
    )
    frame["garch_conditional_vol"] = _fit_garch_conditional_volatility(frame["log_return"])
    frame["vix_level"] = vix_close
    frame["vix_change_5"] = vix_close - vix_close.shift(5)
    # This ratio compares implied volatility with realized volatility to show when option markets are pricing stress ahead of spot moves.
    frame["implied_realized_ratio"] = vix_close / (frame["rolling_std_20"] * np.sqrt(252) * 100)
    frame["atr"] = compute_atr(high, low, close, ATR_PERIOD)
    frame["bb_width"] = compute_bollinger_bands(close, window=20, num_std=2.0)["bb_width"]
    frame["rsi"] = compute_rsi(close, RSI_PERIOD)
    frame["rolling_skew_20"] = frame["log_return"].rolling(window=20).skew()
    frame["rolling_kurt_20"] = frame["log_return"].rolling(window=20).kurt()
    # Volatility acceleration flags capture regimes where today’s realized vol is already outpacing the recent path.
    frame["vol_regime_change"] = (frame["rolling_std_20"] > frame["rolling_std_20"].shift(5)).astype(int)

    feature_frame = _finalize_feature_frame(frame)
    feature_columns = [column for column in feature_frame.columns if column != "Final_Label"]
    logger.info("Built volatility feature frame with %d rows and %d features", len(feature_frame), len(feature_columns))
    return feature_frame


def _cross_asset_frame(spy_df: pd.DataFrame, qqq_df: pd.DataFrame, iwm_df: pd.DataFrame, gld_df: pd.DataFrame, tlt_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Align all market frames on a shared trading calendar."""
    spy_df = _flatten_columns(spy_df)
    qqq_df = _flatten_columns(qqq_df)
    iwm_df = _flatten_columns(iwm_df)
    gld_df = _flatten_columns(gld_df)
    tlt_df = _flatten_columns(tlt_df)
    vix_df = _flatten_columns(vix_df)
    spy = _prepare_frame(spy_df)
    qqq = _prepare_frame(qqq_df)
    iwm = _prepare_frame(iwm_df)
    gld = _prepare_frame(gld_df)
    tlt = _prepare_frame(tlt_df)
    vix = _prepare_frame(vix_df)

    frame = spy.join([
        qqq[["Close"]].rename(columns={"Close": "QQQ_Close"}),
        iwm[["Close"]].rename(columns={"Close": "IWM_Close"}),
        gld[["Close"]].rename(columns={"Close": "GLD_Close"}),
        tlt[["Close"]].rename(columns={"Close": "TLT_Close"}),
        vix[["Close"]].rename(columns={"Close": "VIX_Close"}),
    ], how="inner")
    return frame


def build_bull_bear_features(spy_df: pd.DataFrame, qqq_df: pd.DataFrame, iwm_df: pd.DataFrame, gld_df: pd.DataFrame, tlt_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build bull/bear features using SPY plus cross-asset confirmation signals."""
    spy_df = _ensure_flat_ohlcv(spy_df); qqq_df = _ensure_flat_ohlcv(qqq_df); iwm_df = _ensure_flat_ohlcv(iwm_df); gld_df = _ensure_flat_ohlcv(gld_df); tlt_df = _ensure_flat_ohlcv(tlt_df); vix_df = _ensure_flat_ohlcv(vix_df)
    spy_df = _flatten_columns(spy_df)
    qqq_df = _flatten_columns(qqq_df)
    iwm_df = _flatten_columns(iwm_df)
    gld_df = _flatten_columns(gld_df)
    tlt_df = _flatten_columns(tlt_df)
    vix_df = _flatten_columns(vix_df)
    spy_df = _normalise_ohlcv(spy_df)
    qqq_df = _normalise_ohlcv(qqq_df)
    iwm_df = _normalise_ohlcv(iwm_df)
    gld_df = _normalise_ohlcv(gld_df)
    tlt_df = _normalise_ohlcv(tlt_df)
    vix_df = _normalise_ohlcv(vix_df)

    # Removed due to data leakage with SMA-based labels in experiment phase; using Method B labels instead.
    frame = label_bull_bear_regime(spy_df)
    cross_asset = _cross_asset_frame(spy_df, qqq_df, iwm_df, gld_df, tlt_df, vix_df)
    frame = frame.join(cross_asset[["QQQ_Close", "IWM_Close", "GLD_Close", "TLT_Close", "VIX_Close"]], how="inner")

    close = frame["Close"]
    spy_returns = np.log(close / close.shift(1))

    # Price persistence and range-based signals help the model distinguish broad bull runs from sharp bear market breaks.
    frame["log_return"] = spy_returns
    frame["rolling_mean_5"] = close.rolling(window=5).mean()
    frame["rolling_mean_20"] = close.rolling(window=20).mean()
    frame["rolling_std_20"] = spy_returns.rolling(window=20).std()
    frame["rsi"] = compute_rsi(close, RSI_PERIOD)
    macd = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    frame = frame.join(macd)
    bollinger = compute_bollinger_bands(close, window=20, num_std=2.0)
    frame = frame.join(bollinger[["bb_width"]])
    frame["atr"] = compute_atr(frame["High"], frame["Low"], close, ATR_PERIOD)
    frame["roc_5"] = close / close.shift(5) - 1
    frame["roc_20"] = close / close.shift(20) - 1
    frame["momentum_5"] = close - close.shift(5)
    frame["rolling_skew_20"] = spy_returns.rolling(window=20).skew()
    frame["rolling_kurt_20"] = spy_returns.rolling(window=20).kurt()
    frame["sma_50_ratio"] = close.rolling(window=50).mean() / close

    # Bonds and gold often strengthen when equity markets weaken, so 20-day returns from those assets add useful flight-to-safety context.
    frame["tlt_return_20"] = frame["TLT_Close"] / frame["TLT_Close"].shift(20) - 1
    frame["gld_return_20"] = frame["GLD_Close"] / frame["GLD_Close"].shift(20) - 1
    frame["vix_level"] = frame["VIX_Close"]
    frame["vix_change_10"] = frame["VIX_Close"] - frame["VIX_Close"].shift(10)
    # Small-cap versus large-cap relative strength is a breadth proxy that often weakens before a broad bear move.
    frame["iwm_spy_ratio"] = frame["IWM_Close"] / close
    frame["iwm_spy_momentum"] = frame["iwm_spy_ratio"] - frame["iwm_spy_ratio"].shift(20)
    frame["bond_equity_corr_20"] = spy_returns.rolling(window=20).corr(np.log(frame["TLT_Close"] / frame["TLT_Close"].shift(1)))
    frame["qqq_spy_ratio"] = frame["QQQ_Close"] / close
    frame["qqq_return_20"] = frame["QQQ_Close"] / frame["QQQ_Close"].shift(20) - 1

    feature_frame = _finalize_feature_frame(frame)
    feature_columns = [column for column in feature_frame.columns if column != "Final_Label"]
    logger.info("Built bull/bear feature frame with %d rows and %d features", len(feature_frame), len(feature_columns))
    return feature_frame


def save_baseline_stats(df: pd.DataFrame, regime_type: str) -> dict[str, Any]:
    """Persist mean and standard-deviation baselines for later drift comparison."""
    df = _flatten_columns(df)
    baseline_path = BASELINE_DIR / "feature_baselines.json"
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle) or {}
            if isinstance(loaded, dict):
                existing = loaded

    numeric_columns = [column for column in df.columns if column != "Final_Label" and pd.api.types.is_numeric_dtype(df[column])]
    regime_stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        regime_stats[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
        }

    existing[regime_type] = regime_stats

    with baseline_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)

    logger.info("Saved baseline stats for %s to %s", regime_type, baseline_path)
    return {"path": baseline_path, "stats": regime_stats}


def _label_distribution_text(frame: pd.DataFrame) -> str:
    """Format the label distribution as percentages for reporting."""
    frame = _flatten_columns(frame)
    counts = frame["Final_Label"].value_counts(normalize=True).sort_index()
    return ", ".join(f"{label}={share * 100:.0f}%" for label, share in counts.items())


def _feature_count(frame: pd.DataFrame) -> int:
    """Count only the feature columns, excluding the final label."""
    frame = _flatten_columns(frame)
    return len([column for column in frame.columns if column != "Final_Label"])


def build_features(params: dict[str, Any]) -> None:
    """Backward-compatible entry point used by the command-line runner."""

    _ = params
    logger.info("Feature builders are executed through main(); call the explicit helpers for inference or tests.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feature engineering entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(PARAMS_PATH), help="Path to the project parameter file.")
    return parser.parse_args()


def main() -> None:
    """Run the feature engineering pipeline end-to-end."""

    setup_logging()
    started = pd.Timestamp.now(tz="UTC")
    print("=" * 48)
    print(" MARKET REGIME DETECTION — FEATURE ENGINEERING")
    print(f" Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 48)

    try:
        frames = _load_raw_frames()

        trend_features = build_trend_features(frames["SPY"])
        vol_features = build_vol_features(frames["SPY"], frames["VIX"])
        bull_features = build_bull_bear_features(
            frames["SPY"],
            frames["QQQ"],
            frames["IWM"],
            frames["GLD"],
            frames["TLT"],
            frames["VIX"],
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        trend_path = PROCESSED_DIR / "trend_features.csv"
        vol_path = PROCESSED_DIR / "vol_features.csv"
        bull_path = PROCESSED_DIR / "bull_bear_features.csv"

        trend_features.to_csv(trend_path)
        vol_features.to_csv(vol_path)
        bull_features.to_csv(bull_path)

        save_baseline_stats(trend_features, "trend")
        save_baseline_stats(vol_features, "vol")
        save_baseline_stats(bull_features, "bull_bear")

        print(
            f"[TREND]     Saved trend_features.csv     | Rows: {len(trend_features)} | Features: {_feature_count(trend_features)} | Label: {_label_distribution_text(trend_features)}"
        )
        print(
            f"[VOL]       Saved vol_features.csv       | Rows: {len(vol_features)} | Features: {_feature_count(vol_features)} | Label: {_label_distribution_text(vol_features)}"
        )
        print(
            f"[BULL/BEAR] Saved bull_bear_features.csv | Rows: {len(bull_features)} | Features: {_feature_count(bull_features)} | Label: {_label_distribution_text(bull_features)}"
        )
        print("Next step: python src/train.py")
        print("=" * 48)
    except Exception as exc:
        logger.exception("Feature engineering failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()