"""Create the Market Regime Detection project scaffold.

This script is intentionally idempotent: it creates the directory structure,
adds the package marker files, and initializes the prediction log without
overwriting any existing work.
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DIRECTORIES = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "baselines",
    PROJECT_ROOT / "data" / "simulation",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "api",
    PROJECT_ROOT / "frontend",
    PROJECT_ROOT / "airflow_dags",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "prometheus",
    PROJECT_ROOT / "grafana",
    PROJECT_ROOT / "grafana" / "dashboards",
]

INIT_FILES = [
    PROJECT_ROOT / "src" / "__init__.py",
    PROJECT_ROOT / "api" / "__init__.py",
    PROJECT_ROOT / "tests" / "__init__.py",
]

PREDICTION_LOG = PROJECT_ROOT / "data" / "simulation" / "prediction_log.csv"


def ensure_directory(path: Path) -> Path:
    """Create a directory and all required parents if they do not exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_empty_file(path: Path) -> Path:
    """Create an empty file only when it is missing."""

    path.touch(exist_ok=True)
    return path


def ensure_prediction_log(path: Path) -> Path:
    """Create the live prediction log with the requested header row."""

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "ticker",
                    "regime_type",
                    "predicted_label",
                    "confidence",
                    "features_snapshot",
                ]
            )
    return path


def print_tree(created_paths: list[Path]) -> None:
    """Print a compact confirmation tree for the scaffolded items."""

    print("Created project scaffold:")
    for path in created_paths:
        relative_path = path.relative_to(PROJECT_ROOT)
        suffix = "/" if path.is_dir() else ""
        print(f"- {relative_path.as_posix()}{suffix}")


def main() -> None:
    created_paths: list[Path] = []

    for directory in DIRECTORIES:
        created_paths.append(ensure_directory(directory))

    for init_file in INIT_FILES:
        created_paths.append(ensure_empty_file(init_file))

    created_paths.append(ensure_prediction_log(PREDICTION_LOG))

    print_tree(created_paths)
    print("Run `git init && dvc init` after this script completes")
    print("Next step: run python src/data_ingestion.py")


if __name__ == "__main__":
    main()