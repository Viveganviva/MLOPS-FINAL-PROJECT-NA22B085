"""Export DVC metrics and plot CSVs from the trained regime models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
TRAINING_REPORT_PATH = PROJECT_ROOT / "data" / "baselines" / "training_report.json"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_PATH = PROJECT_ROOT / "dvc_metrics.json"
PLOTS_DIR = PROJECT_ROOT / "dvc_plots"


def _load_params() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    with PARAMS_PATH.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    if not isinstance(params, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")
    return params


PARAMS = _load_params()


def _load_report() -> dict[str, Any]:
    if not TRAINING_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing training report: {TRAINING_REPORT_PATH}")
    with TRAINING_REPORT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def _load_frame(regime_type: str) -> pd.DataFrame:
    processed_path = PROCESSED_DIR / f"{regime_type}_features.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed features: {processed_path}")
    frame = pd.read_csv(processed_path, parse_dates=[0], index_col=0)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_split = pd.Timestamp(PARAMS["training"]["val_split_date"])
    test_split = pd.Timestamp(PARAMS["training"]["test_split_date"])
    train = frame.loc[frame.index < val_split]
    val = frame.loc[(frame.index >= val_split) & (frame.index < test_split)]
    test = frame.loc[frame.index >= test_split]
    return train, val, test


def _load_artifacts(regime_type: str) -> tuple[Any, Any, list[str], list[str]]:
    model = joblib.load(MODELS_DIR / f"{regime_type}_model.pkl")
    scaler = joblib.load(MODELS_DIR / f"{regime_type}_scaler.pkl")
    with (MODELS_DIR / f"{regime_type}_feature_columns.json").open("r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    with (MODELS_DIR / f"{regime_type}_classes.json").open("r", encoding="utf-8") as handle:
        class_labels = json.load(handle)
    return model, scaler, feature_columns, class_labels


def _prepare_xy(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x_frame = frame.reindex(columns=feature_columns).copy().fillna(0.0)
    y_series = frame["Final_Label"].astype(str).copy()
    x_frame = x_frame.dropna()
    y_series = y_series.loc[x_frame.index]
    return x_frame, y_series


def _predict_frame(model: Any, scaler: Any, x_frame: pd.DataFrame, class_labels: list[str]) -> list[str]:
    scaled = scaler.transform(x_frame)
    predictions = model.predict(scaled)
    result: list[str] = []
    for prediction in predictions:
        if isinstance(prediction, (int, np.integer)) and class_labels and int(prediction) < len(class_labels):
            result.append(str(class_labels[int(prediction)]))
        else:
            result.append(str(prediction))
    return result


def _write_confusion_csv(regime_type: str, actual: list[str], predicted: list[str]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    frame.to_csv(PLOTS_DIR / f"confusion_matrix_{regime_type}.csv", index=False)


def main() -> None:
    report = _load_report()
    metrics: dict[str, dict[str, float]] = {}

    for result in report.get("results", []):
        regime_type = result["regime_type"]
        frame = _load_frame(regime_type)
        _, _, test_frame = _split_frame(frame)
        model, scaler, feature_columns, class_labels = _load_artifacts(regime_type)
        x_test, y_test = _prepare_xy(test_frame, feature_columns)
        predicted = _predict_frame(model, scaler, x_test, class_labels)

        _write_confusion_csv(regime_type, y_test.tolist(), predicted)
        metrics[regime_type] = {
            "test_f1": float(f1_score(y_test, predicted, average="weighted")),
            "test_accuracy": float(accuracy_score(y_test, predicted)),
        }

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("DVC metrics saved. Run `dvc metrics show` to view. Run `dvc plots show` for visualizations.")


if __name__ == "__main__":
    main()