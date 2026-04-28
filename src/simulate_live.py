"""Replay held-out market dates against the API to simulate a live stream."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_LOG = PROJECT_ROOT / "data" / "simulation" / "prediction_log.csv"


@dataclass(slots=True)
class SimulationStats:
    """Collect simple counters for the replay summary."""

    dates_processed: int = 0
    prediction_batches: int = 0
    predictions_sent: int = 0
    drift_checks: int = 0
    retraining_triggers: int = 0
    failures: int = 0


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL for the running API service.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol to replay.")
    parser.add_argument("--start-date", default="2023-01-01", help="First simulation date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2024-12-31", help="Last simulation date (YYYY-MM-DD).")
    parser.add_argument("--speed", type=float, default=0.5, help="Pause between simulated trading days in seconds.")
    parser.add_argument(
        "--regimes",
        default="trend,vol,bull_bear",
        help="Comma-separated regime types to request from the API.",
    )
    parser.add_argument(
        "--drift-every-n-days",
        type=int,
        default=5,
        help="Run drift checks every N simulated trading days.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def _parse_regimes(regimes: str) -> list[str]:
    parsed = [item.strip() for item in regimes.split(",") if item.strip()]
    return parsed or ["trend", "vol", "bull_bear"]


def _ensure_prediction_log() -> None:
    """Make sure the API-managed prediction log exists before replay starts."""

    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PREDICTION_LOG.exists():
        PREDICTION_LOG.write_text(
            "timestamp,simulation_date,ticker,regime_type,predicted_label,confidence,inference_latency_ms,features_snapshot\n",
            encoding="utf-8",
        )


def _check_api_ready(session: requests.Session, api_url: str, timeout: float) -> None:
    """Fail fast if the API is not ready for simulation."""

    health = session.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
    health.raise_for_status()
    payload = health.json()
    if payload.get("status") not in {"ok", "degraded"}:
        raise RuntimeError(f"API health check failed: {payload}")

    ready = session.get(f"{api_url.rstrip('/')}/ready", timeout=timeout)
    ready.raise_for_status()
    ready_payload = ready.json()
    if not ready_payload.get("ready", False):
        raise RuntimeError(f"API readiness check failed: {ready_payload}")


def _predict_day(
    session: requests.Session,
    api_url: str,
    ticker: str,
    sim_date: str,
    regimes: list[str],
    timeout: float,
) -> dict[str, Any]:
    """Request predictions for one simulation date."""

    response = session.post(
        f"{api_url.rstrip('/')}/predict",
        json={"ticker": ticker, "as_of_date": sim_date, "regime_types": regimes},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _check_drift(session: requests.Session, api_url: str, regimes: list[str], timeout: float) -> tuple[bool, list[dict[str, Any]]]:
    """Run drift checks for all regimes and trigger retraining if needed."""

    drift_reports: list[dict[str, Any]] = []
    retrain_needed = False

    for regime_type in regimes:
        response = session.get(f"{api_url.rstrip('/')}/drift/{regime_type}", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        drift_reports.append(payload)
        retrain_needed = retrain_needed or bool(payload.get("retrain_recommended"))

    return retrain_needed, drift_reports


def _trigger_retrain(session: requests.Session, api_url: str, reason: str, timeout: float) -> dict[str, Any]:
    """Start retraining via the API and return the accepted payload."""

    response = session.post(f"{api_url.rstrip('/')}/retrain", params={"triggered_by": reason}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _print_prediction_summary(sim_date: str, result: dict[str, Any]) -> None:
    """Emit one compact console line for the current replay date."""

    pieces: list[str] = []
    for regime_type, payload in result["results"].items():
        pieces.append(
            f"{regime_type}={payload['predicted_label']} ({payload['confidence']:.3f}, {payload['inference_latency_ms']:.1f} ms)"
        )
    print(f"[{sim_date}] {result['ticker']} | " + " | ".join(pieces))


def simulate(api_url: str, ticker: str, start_date: str, end_date: str, speed: float, regimes: list[str], drift_every_n_days: int, timeout: float) -> SimulationStats:
    """Replay a business-day range against the API and keep simple summary counters."""

    stats = SimulationStats()
    _ensure_prediction_log()

    with requests.Session() as session:
        _check_api_ready(session, api_url, timeout)
        dates = pd.bdate_range(start=start_date, end=end_date)

        print("=" * 78)
        print(" MARKET REGIME LIVE SIMULATION ")
        print("=" * 78)
        print(f"Ticker: {ticker}")
        print(f"Date range: {start_date} -> {end_date}")
        print(f"API: {api_url}")
        print(f"Regimes: {', '.join(regimes)}")
        print(f"Drift check cadence: every {drift_every_n_days} simulated trading days")
        print("=" * 78)

        for index, date in enumerate(dates, start=1):
            sim_date = date.strftime("%Y-%m-%d")
            try:
                payload = _predict_day(session, api_url, ticker, sim_date, regimes, timeout)
                stats.prediction_batches += 1
                stats.predictions_sent += len(payload.get("results", {}))
                stats.dates_processed += 1
                _print_prediction_summary(sim_date, payload)
            except Exception as exc:
                stats.failures += 1
                LOGGER.exception("Prediction failed for %s", sim_date)
                print(f"[{sim_date}] prediction failed: {exc}")
                continue

            if drift_every_n_days > 0 and index % drift_every_n_days == 0:
                try:
                    retrain_needed, drift_reports = _check_drift(session, api_url, regimes, timeout)
                    stats.drift_checks += len(drift_reports)
                    if retrain_needed:
                        retrain_payload = _trigger_retrain(session, api_url, reason="auto_drift", timeout=timeout)
                        stats.retraining_triggers += 1
                        print(f"[{sim_date}] retraining triggered: {retrain_payload.get('message', retrain_payload)}")
                    else:
                        print(f"[{sim_date}] drift check passed")
                except Exception as exc:
                    stats.failures += 1
                    LOGGER.exception("Drift check failed for %s", sim_date)
                    print(f"[{sim_date}] drift check failed: {exc}")

            if speed > 0:
                time.sleep(speed)

    return stats


def main() -> None:
    configure_logging()
    args = parse_args()
    regimes = _parse_regimes(args.regimes)

    try:
        stats = simulate(
            api_url=args.api_url,
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            speed=max(args.speed, 0.0),
            regimes=regimes,
            drift_every_n_days=max(args.drift_every_n_days, 0),
            timeout=args.timeout,
        )
        print("=" * 78)
        print(" SIMULATION SUMMARY ")
        print("=" * 78)
        print(f"Dates processed: {stats.dates_processed}")
        print(f"Prediction batches: {stats.prediction_batches}")
        print(f"Individual predictions: {stats.predictions_sent}")
        print(f"Drift checks: {stats.drift_checks}")
        print(f"Retraining triggers: {stats.retraining_triggers}")
        print(f"Failures: {stats.failures}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as exc:
        LOGGER.exception("Simulation failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
