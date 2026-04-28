"""Train the three regime models and register them with MLflow when available."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:  # pragma: no cover - optional dependency guard
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - runtime fallback when MLflow is unavailable
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - runtime fallback when xgboost is unavailable
    XGBClassifier = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    """Load the centralized project parameters."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load params.yaml.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("params.yaml must contain a mapping at the top level.")
    return config


PARAMS = load_params()
TRAINING_PARAMS = PARAMS["training"]
MLFLOW_PARAMS = PARAMS["mlflow"]
BASELINE_DIR = PROJECT_ROOT / PARAMS["data"]["baseline_dir"]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_ENGINEERING_PATHS = {
    "trend": PROJECT_ROOT / PARAMS["data"]["processed_dir"] / "trend_features.csv",
    "vol": PROJECT_ROOT / PARAMS["data"]["processed_dir"] / "vol_features.csv",
    "bull_bear": PROJECT_ROOT / PARAMS["data"]["processed_dir"] / "bull_bear_features.csv",
}


logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console and file logging for the training run."""

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "training.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def models_already_deployed() -> bool:
    """Check whether all three model artifacts exist and are younger than seven days."""

    model_files = [MODELS_DIR / "trend_model.pkl", MODELS_DIR / "vol_model.pkl", MODELS_DIR / "bull_bear_model.pkl"]
    if all(path.exists() for path in model_files):
        oldest = min(path.stat().st_mtime for path in model_files)
        age_days = (time.time() - oldest) / 86400.0
        if age_days < 7:
            logger.info("Models already exist and are %.1f days old. Use --force to retrain.", age_days)
            return True
    return False


def setup_mlflow(params: dict[str, Any]) -> None:
    """
    Set up MLflow tracking. Tries the Docker server first (for when running inside compose),
    then falls back to local file store for standalone development runs.
    The tracking URI is also set via environment variable so Docker can override it.
    """

    if mlflow is None:
        logger.warning("MLflow is not installed in the active environment; logging will be skipped.")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        logger.info("Connected to MLflow server at: %s", tracking_uri)
    except Exception as exc:
        fallback_uri = "./mlruns"
        logger.warning("MLflow server not reachable (%s). Falling back to local store: %s", exc, fallback_uri)
        mlflow.set_tracking_uri(fallback_uri)


def _load_feature_frame(features_path: Path, label_col: str) -> pd.DataFrame:
    """Load a processed feature frame and sort it by time."""

    frame = pd.read_csv(features_path, index_col=0, parse_dates=[0])
    if label_col not in frame.columns:
        raise KeyError(f"Expected label column '{label_col}' in {features_path}")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame


def _time_split(frame: pd.DataFrame, label_col: str, val_split_date: str, test_split_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a time-indexed frame into train, validation, and test partitions."""

    train = frame.loc[frame.index < pd.Timestamp(val_split_date)]
    val = frame.loc[(frame.index >= pd.Timestamp(val_split_date)) & (frame.index < pd.Timestamp(test_split_date))]
    test = frame.loc[frame.index >= pd.Timestamp(test_split_date)]

    feature_columns = [column for column in frame.columns if column != label_col]
    x_train = train[feature_columns]
    x_val = val[feature_columns]
    x_test = test[feature_columns]
    y_train = train[label_col]
    y_val = val[label_col]
    y_test = test[label_col]
    return x_train, x_val, x_test, y_train, y_val, y_test


def _prepare_target_encoder(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray]:
    """Encode string labels into integer classes shared across all splits."""

    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_val, y_test], axis=0).astype(str))
    return encoder, encoder.transform(y_train.astype(str)), encoder.transform(y_val.astype(str)), encoder.transform(y_test.astype(str))


def _build_estimator(model_params: dict[str, Any]) -> Any:
    """Create the primary classifier, using XGBoost when available and a tree fallback otherwise."""

    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=int(model_params["n_estimators"]),
            max_depth=int(model_params["max_depth"]),
            learning_rate=float(model_params["learning_rate"]),
            subsample=float(model_params["subsample"]),
            min_child_weight=int(model_params.get("min_child_weight", 1)),
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=int(TRAINING_PARAMS["random_state"]),
            tree_method="hist",
        )

    logger.warning("xgboost is unavailable; falling back to RandomForestClassifier for training.")
    return RandomForestClassifier(
        n_estimators=int(model_params["n_estimators"]),
        max_depth=int(model_params["max_depth"]),
        random_state=int(TRAINING_PARAMS["random_state"]),
    )


def _feature_importance_values(model: Any, feature_names: list[str]) -> np.ndarray:
    """Return normalized feature importance values when available."""

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        if values.sum() > 0:
            return values / values.sum()
        return values
    if hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=float))
        values = coef.mean(axis=0)
        return values / values.sum() if values.sum() > 0 else values
    return np.zeros(len(feature_names), dtype=float)


def _positive_class_probability(model: Any, x_frame: pd.DataFrame) -> np.ndarray:
    """Return the probability associated with the positive or most likely class."""

    probabilities = model.predict_proba(x_frame)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    return probabilities


def _binary_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Compute AUC for binary or multiclass outputs with a safe fallback."""

    try:
        if len(np.unique(y_true)) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def _evaluate_split(model: Any, x_frame: pd.DataFrame, y_true: np.ndarray) -> dict[str, float]:
    """Evaluate a fitted classifier on a given split."""

    y_pred = model.predict(x_frame)
    proba = _positive_class_probability(model, x_frame)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc": _binary_auc(y_true, proba),
    }


def _format_comparison_table(regime_type: str, xgb_metrics: dict[str, float], lr_metrics: dict[str, float]) -> None:
    """Print a compact comparison between the baseline and primary model."""

    print(f"\n{regime_type.upper()} MODEL COMPARISON")
    print("Metric           Logistic Regression   XGBoost/Fallback")
    print(f"Val F1           {lr_metrics['val_f1']:.3f}                 {xgb_metrics['val_f1']:.3f}")
    print(f"Val Acc          {lr_metrics['val_accuracy']:.3f}                 {xgb_metrics['val_accuracy']:.3f}")
    print(f"Test F1          {lr_metrics['test_f1']:.3f}                 {xgb_metrics['test_f1']:.3f}")
    print(f"Test Acc         {lr_metrics['test_accuracy']:.3f}                 {xgb_metrics['test_accuracy']:.3f}")
    print(f"Test Precision   {lr_metrics['test_precision']:.3f}                 {xgb_metrics['test_precision']:.3f}")
    print(f"Test Recall      {lr_metrics['test_recall']:.3f}                 {xgb_metrics['test_recall']:.3f}")
    print(f"Test AUC         {lr_metrics['test_auc']:.3f}                 {xgb_metrics['test_auc']:.3f}")


def _save_plots(regime_type: str, model: Any, feature_names: list[str], x_test: pd.DataFrame, y_test: np.ndarray) -> tuple[Path, Path]:
    """Create feature-importance and confusion-matrix plots for MLflow artifacts."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    importance = _feature_importance_values(model, feature_names)
    importance_path = MODELS_DIR / f"{regime_type}_feature_importance.png"
    confusion_path = MODELS_DIR / f"{regime_type}_confusion_matrix.png"

    order = np.argsort(importance)[::-1][: min(len(feature_names), 20)]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[order], y=[feature_names[i] for i in order], color="#4c78a8")
    plt.title(f"{regime_type.title()} Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=200)
    plt.close()

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{regime_type.title()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=200)
    plt.close()

    return importance_path, confusion_path


def _log_to_mlflow(run, regime_type: str, model: Any, scaler: StandardScaler, importance_path: Path, confusion_path: Path, model_name: str, feature_columns: list[str], metrics: dict[str, float], extra_params: dict[str, Any]) -> tuple[str, Any]:
    """Log parameters, metrics, artifacts, and model registry entries to MLflow when available."""

    run_id = run.info.run_id
    model_version = None

    if mlflow is None:
        return run_id, model_version

    mlflow.set_tag("regime_type", regime_type)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        git_commit = "unknown"
    mlflow.set_tag("git_commit", git_commit)
    mlflow.log_params({**extra_params, "feature_count": len(feature_columns)})
    mlflow.log_metrics(metrics)
    mlflow.log_metric("train_test_f1_gap", abs(metrics.get("train_f1", 0.0) - metrics.get("test_f1", 0.0)))
    mlflow.log_artifact(str(importance_path))
    mlflow.log_artifact(str(confusion_path))
    mlflow.sklearn.log_model(model, artifact_path=f"{regime_type}_model", registered_model_name=model_name)

    scaler_path = MODELS_DIR / f"{regime_type}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(str(scaler_path))

    feature_path = MODELS_DIR / f"{regime_type}_feature_columns.json"
    with feature_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)
    mlflow.log_artifact(str(feature_path))

    if MlflowClient is not None:
        try:
            client = MlflowClient()
            latest = client.get_latest_versions(model_name, stages=["None"])
            if latest:
                client.transition_model_version_stage(model_name, latest[0].version, "Production")
                logger.info("Model %s v%s -> Production", model_name, latest[0].version)
                model_version = latest[0].version
        except Exception as exc:
            logger.warning("Could not transition model %s to Production: %s", model_name, exc)

    return run_id, model_version


def train_single_regime(regime_type: str, features_path: Path, label_col: str, experiment_name: str, model_name: str, model_params: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate a single regime model end to end."""

    frame = _load_feature_frame(features_path, label_col)
    logger.info("[%s] Loaded feature frame %s", regime_type, frame.shape)
    logger.info("[%s] Label distribution: %s", regime_type, frame[label_col].value_counts(dropna=False).to_dict())

    x_train, x_val, x_test, y_train, y_val, y_test = _time_split(
        frame,
        label_col,
        TRAINING_PARAMS["val_split_date"],
        TRAINING_PARAMS["test_split_date"],
    )

    print(f"[{regime_type.upper()}] Train/Val/Test sizes: {len(x_train)}/{len(x_val)}/{len(x_test)}")

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_val_scaled = pd.DataFrame(scaler.transform(x_val), index=x_val.index, columns=x_val.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    encoder, y_train_enc, y_val_enc, y_test_enc = _prepare_target_encoder(y_train, y_val, y_test)
    classes_path = MODELS_DIR / f"{regime_type}_classes.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with classes_path.open("w", encoding="utf-8") as handle:
        json.dump(encoder.classes_.tolist(), handle, indent=2)

    xgb_model = _build_estimator(model_params)
    xgb_model.fit(x_train_scaled, y_train_enc)

    lr_model = LogisticRegression(max_iter=2000, multi_class="auto", random_state=int(TRAINING_PARAMS["random_state"]))
    lr_model.fit(x_train_scaled, y_train_enc)

    xgb_train_metrics = _evaluate_split(xgb_model, x_train_scaled, y_train_enc)
    xgb_val_metrics = _evaluate_split(xgb_model, x_val_scaled, y_val_enc)
    xgb_test_metrics = _evaluate_split(xgb_model, x_test_scaled, y_test_enc)

    lr_val_metrics = _evaluate_split(lr_model, x_val_scaled, y_val_enc)
    lr_test_metrics = _evaluate_split(lr_model, x_test_scaled, y_test_enc)

    _format_comparison_table(
        regime_type,
        {"val_f1": xgb_val_metrics["f1"], "test_f1": xgb_test_metrics["f1"], "test_accuracy": xgb_test_metrics["accuracy"], "test_precision": xgb_test_metrics["precision"], "test_recall": xgb_test_metrics["recall"], "test_auc": xgb_test_metrics["auc"]},
        {"val_f1": lr_val_metrics["f1"], "val_accuracy": lr_val_metrics["accuracy"], "test_f1": lr_test_metrics["f1"], "test_accuracy": lr_test_metrics["accuracy"], "test_precision": lr_test_metrics["precision"], "test_recall": lr_test_metrics["recall"], "test_auc": lr_test_metrics["auc"]},
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    importance_path, confusion_path = _save_plots(regime_type, xgb_model, list(x_train.columns), x_test_scaled, y_test_enc)

    extra_params = {
        **{key: value for key, value in model_params.items()},
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "train_start": str(x_train.index.min()),
        "train_end": str(x_train.index.max()),
        "val_start": str(x_val.index.min()),
        "val_end": str(x_val.index.max()),
        "test_start": str(x_test.index.min()),
        "test_end": str(x_test.index.max()),
    }

    run_id = "local"
    model_version = None

    if mlflow is not None:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{regime_type}_{timestamp}") as run:
            run_id = run.info.run_id
            run_id, model_version = _log_to_mlflow(
                run,
                regime_type,
                xgb_model,
                scaler,
                importance_path,
                confusion_path,
                model_name,
                list(x_train.columns),
                {
                    "val_accuracy": xgb_val_metrics["accuracy"],
                    "val_f1": xgb_val_metrics["f1"],
                    "test_accuracy": xgb_test_metrics["accuracy"],
                    "test_f1": xgb_test_metrics["f1"],
                    "test_precision": xgb_test_metrics["precision"],
                    "test_recall": xgb_test_metrics["recall"],
                    "test_auc": xgb_test_metrics["auc"],
                    "baseline_lr_f1": lr_test_metrics["f1"],
                    "train_f1": xgb_train_metrics["f1"],
                },
                extra_params,
            )
            logger.info("MLflow run ID: %s", run.info.run_id)
            mlflow.log_artifact(str(classes_path))
    else:
        scaler_path = MODELS_DIR / f"{regime_type}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        with (MODELS_DIR / f"{regime_type}_feature_columns.json").open("w", encoding="utf-8") as handle:
            json.dump(list(x_train.columns), handle, indent=2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODELS_DIR / f"{regime_type}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{regime_type}_scaler.pkl")

    return {
        "regime_type": regime_type,
        "test_f1": xgb_test_metrics["f1"],
        "test_accuracy": xgb_test_metrics["accuracy"],
        "run_id": run_id,
        "model_version": model_version,
    }


def save_training_report(results_list: list[dict[str, Any]]) -> Path:
    """Save a compact training report for downstream monitoring and retraining decisions."""

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    report_path = BASELINE_DIR / "training_report.json"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results_list,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("╔══════════════════╦══════════════╦════════════╦═════════════╗")
    print("║ Regime           ║ Test F1      ║ Test Acc   ║ MLflow RunID║")
    print("╠══════════════════╬══════════════╬════════════╬═════════════╣")
    for result in results_list:
        display = {
            "trend": "Trend/MeanRev",
            "vol": "Volatility",
            "bull_bear": "Bull/Bear",
        }.get(result["regime_type"], result["regime_type"])
        run_id = str(result["run_id"])[:11]
        print(f"║ {display:<16} ║ {result['test_f1']:<12.2f} ║ {result['test_accuracy']:<10.2f} ║ {run_id:<11} ║")
    print("╚══════════════════╩══════════════╩════════════╩═════════════╝")
    return report_path


def train_models(params: dict[str, Any], force: bool = False) -> list[dict[str, Any]]:
    """Train all three regime models and return their result summaries."""

    if not force and models_already_deployed():
        logger.info("Existing models are fresh enough. Use --force to retrain.")
        return []

    setup_mlflow(params)

    results = [
        train_single_regime("trend", FEATURE_ENGINEERING_PATHS["trend"], "Final_Label", params["mlflow"]["experiment_trend"], params["mlflow"]["model_names"]["trend"], params["training"]["trend_model"]),
        train_single_regime("vol", FEATURE_ENGINEERING_PATHS["vol"], "Final_Label", params["mlflow"]["experiment_vol"], params["mlflow"]["model_names"]["vol"], params["training"]["vol_model"]),
        train_single_regime("bull_bear", FEATURE_ENGINEERING_PATHS["bull_bear"], "Final_Label", params["mlflow"]["experiment_bull_bear"], params["mlflow"]["model_names"]["bull_bear"], params["training"]["bull_bear_model"]),
    ]
    save_training_report(results)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the training entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(PARAMS_PATH), help="Path to the project parameter file.")
    parser.add_argument("--force", action="store_true", help="Retrain even if existing model artifacts are fresh.")
    return parser.parse_args()


def main(force: bool | None = None) -> list[dict[str, Any]]:
    """Run training, either from CLI or from another module."""

    setup_logging()
    params = PARAMS

    if force is None:
        args = _parse_args()
        force = args.force

    if not force and models_already_deployed():
        print("Models already exist and are recent. Use --force to retrain.")
        return []

    results = train_models(params, force=bool(force))
    print("Next step: docker-compose up --build")
    return results


if __name__ == "__main__":
    main()