"""Ingest historical price data for the market regime pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent logging format for command-line execution."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_params(path: Path | str = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from params.yaml."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")

    return config


def ingest_data(params: dict[str, Any]) -> None:
    """Placeholder ingestion routine that will later download market data."""

    data_config = params.get("data", {})
    raw_dir = Path(data_config.get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Ingestion scaffold ready for tickers: %s", data_config.get("train_tickers", []))
    LOGGER.info("Raw data directory confirmed at %s", raw_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="Path to the project parameter file.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        params = load_params(args.params)
        ingest_data(params)
    except Exception as exc:  # pragma: no cover - command-line safety net
        LOGGER.exception("Data ingestion failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()