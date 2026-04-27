"""Feature engineering entry point for regime modeling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_params(path: Path | str = "params.yaml") -> dict[str, Any]:
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


def build_features(params: dict[str, Any]) -> None:
    """Placeholder feature builder; the production implementation will expand this."""

    feature_config = params.get("features", {})
    processed_dir = Path(params.get("data", {}).get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Feature windows configured: short=%s, mid=%s, long=%s", feature_config.get("short_window"), feature_config.get("mid_window"), feature_config.get("long_window"))
    LOGGER.info("Processed data directory confirmed at %s", processed_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="Path to the project parameter file.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        params = load_params(args.params)
        build_features(params)
    except Exception as exc:  # pragma: no cover - command-line safety net
        LOGGER.exception("Feature engineering failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()