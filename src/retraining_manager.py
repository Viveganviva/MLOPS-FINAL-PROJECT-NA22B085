"""Retraining orchestration for the market regime pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import data_ingestion, feature_engineering, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
LOG_FILE = PROJECT_ROOT / "logs" / "retraining.log"


def _load_params() -> dict[str, Any]:
    """Load the shared parameter file used by the retraining workflow."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    with PARAMS_PATH.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")
    return params


class RetrainingManager:
    """Coordinate the retraining workflow behind a simple file lock."""

    def __init__(self) -> None:
        self.params = _load_params()
        self.lock_file = PROJECT_ROOT / "data" / "baselines" / ".retrain_lock"
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Create a dedicated retraining logger that writes to logs/retraining.log."""

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"{__name__}.RetrainingManager")
        logger.setLevel(logging.INFO)
        if not any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == str(LOG_FILE) for handler in logger.handlers):
            handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(handler)
        return logger

    def is_retrain_locked(self) -> bool:
        """Return True when a fresh retrain lock exists."""

        if not self.lock_file.exists():
            return False

        age_minutes = (time.time() - self.lock_file.stat().st_mtime) / 60.0
        if age_minutes > 60:
            try:
                self.lock_file.unlink()
            except Exception as exc:
                self.logger.warning("Failed to remove stale retrain lock: %s", exc)
            return False
        return True

    def acquire_lock(self) -> None:
        """Create a lock file containing the PID and timestamp."""

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "timestamp": datetime.now(timezone.utc).isoformat()}
        with self.lock_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def release_lock(self) -> None:
        """Remove the retraining lock file if it exists."""

        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except Exception as exc:
                self.logger.warning("Failed to release retrain lock cleanly: %s", exc)

    def _update_training_report_timestamp(self) -> None:
        """Refresh the training report timestamp after a retrain completes."""

        report_path = PROJECT_ROOT / "data" / "baselines" / "training_report.json"
        report = {"timestamp": datetime.now(timezone.utc).isoformat(), "results": []}
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle) or {}
                if isinstance(existing, dict):
                    report["results"] = existing.get("results", [])
            except Exception:
                pass
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    def run_retrain_pipeline(self, triggered_by: str = "manual") -> dict[str, Any]:
        """Run the full retraining pipeline with lock protection and cleanup."""

        if self.is_retrain_locked():
            self.logger.warning("[RETRAIN] Pipeline skipped because lock is active")
            return {"status": "skipped", "reason": "lock active"}

        self.acquire_lock()
        start_time = time.time()
        self.logger.info("[RETRAIN] Pipeline triggered by: %s", triggered_by)

        try:
            data_ingestion.download_training_data()
            feature_engineering.main()

            # For retraining runs, we override split dates via env vars so the model
            # always uses the most recent data for validation and testing regardless
            # of how much new data has been accumulated since initial training.
            # The train module reads these env vars and passes them to _time_split().
            os.environ['RETRAIN_VAL_SPLIT'] = 'last_20pct'
            os.environ['RETRAIN_TEST_SPLIT'] = 'last_10pct'

            train.main(force=True)
            self._update_training_report_timestamp()

            elapsed = time.time() - start_time
            self.logger.info("[RETRAIN] Pipeline complete. Duration: %.1f seconds", elapsed)
            return {"status": "complete", "duration_seconds": elapsed, "triggered_by": triggered_by}
        except Exception as exc:
            self.logger.exception("[RETRAIN] Pipeline failed: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            self.release_lock()

    def run_retrain_async(self, triggered_by: str = "auto_drift") -> dict[str, Any]:
        """Start retraining in a background thread and return immediately."""

        thread = threading.Thread(target=self.run_retrain_pipeline, kwargs={"triggered_by": triggered_by}, daemon=True)
        thread.start()
        return {"status": "started", "thread_id": thread.ident}


_DEFAULT_MANAGER: RetrainingManager | None = None


def _get_manager() -> RetrainingManager:
    """Return a module-level retraining manager instance."""

    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = RetrainingManager()
    return _DEFAULT_MANAGER


def run_retrain_pipeline(triggered_by: str = "manual") -> dict[str, Any]:
    """Convenience wrapper for callers that expect a module-level function."""

    return _get_manager().run_retrain_pipeline(triggered_by=triggered_by)


def run_retrain_async(triggered_by: str = "auto_drift") -> dict[str, Any]:
    """Convenience wrapper for callers that expect a module-level async starter."""

    return _get_manager().run_retrain_async(triggered_by=triggered_by)


def _parse_args() -> argparse.Namespace:
    """Parse the command-line interface for manual retraining."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triggered-by", default="manual", help="Reason or source that triggered retraining.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manager = RetrainingManager()
    print(manager.run_retrain_pipeline(triggered_by=args.triggered_by))