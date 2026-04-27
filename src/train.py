"""Train regime detection models from engineered features."""

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

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")

    return config


def train_models(params: dict[str, Any], n_estimators: int | None = None, max_depth: int | None = None) -> None:
    """Placeholder trainer that records the intended model configuration."""

    training_config = params.get("training", {})
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Training scaffold ready with random_state=%s", training_config.get("random_state"))
    LOGGER.info("CLI overrides received: n_estimators=%s, max_depth=%s", n_estimators, max_depth)
    LOGGER.info("Model artifacts will be written to %s", models_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="Path to the project parameter file.")
    parser.add_argument("--n-estimators", type=int, default=None, help="Optional override for tree-based model size.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional override for tree-based model depth.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        params = load_params(args.params)
        train_models(params, n_estimators=args.n_estimators, max_depth=args.max_depth)
    except Exception as exc:  # pragma: no cover - command-line safety net
        LOGGER.exception("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()