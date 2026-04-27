"""Simulate a live prediction stream for the held-out market window."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def simulate(speed: float = 2.0) -> Path:
    """Prepare the simulation directory and return the prediction log path."""

    simulation_dir = Path("data/simulation")
    simulation_dir.mkdir(parents=True, exist_ok=True)
    log_path = simulation_dir / "prediction_log.csv"
    LOGGER.info("Simulation scaffold active at speed=%s seconds per step", speed)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=2.0, help="Seconds between simulated live updates.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    try:
        log_path = simulate(speed=args.speed)
        LOGGER.info("Live prediction log available at %s", log_path)
    except Exception as exc:  # pragma: no cover - command-line safety net
        LOGGER.exception("Simulation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()