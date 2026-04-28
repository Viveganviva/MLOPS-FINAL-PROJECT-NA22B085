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

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except ImportError as _mlflow_err:
    import sys, traceback
    print(f"[ERROR] MLflow import failed: {_mlflow_err}", file=sys.stderr)
    traceback.print_exc()
    mlflow = None
    MlflowClient = None


try:
    from xgboost import XGBClassifier
except ImportError as _xgb_err:
    import sys, traceback
    print(f"[ERROR] XGBoost import failed: {_xgb_err}", file=sys.stderr)
    traceback.print_exc()
    XGBClassifier = None

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
        logger.warning("MLflow is not available. Skipping experiment tracking.")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    try:
        # Lightweight ping — just create/get the default experiment
        mlflow.set_experiment("__connection_test__")
        mlflow.set_experiment(params["mlflow"]["experiment_trend"])
        logger.info("Connected to MLflow server at: %s", tracking_uri)
    except Exception as exc:
        fallback_uri = "./mlruns"
        logger.warning("MLflow server not reachable (%s). Falling back to: %s", exc, fallback_uri)
        mlflow.set_tracking_uri(fallback_uri)


def _load_feature_frame(features_path: Path, label_col: str) -> pd.DataFrame:
    """Load a processed feature frame and sort it by time."""

    frame = pd.read_csv(features_path, index_col=0, parse_dates=[0])
    if label_col not in frame.columns:
        raise KeyError(f"Expected label column '{label_col}' in {features_path}")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame


def _time_split(
    frame: pd.DataFrame,
    label_col: str,
    val_split_date: str,
    test_split_date: str,
) -> tuple:
    """
    Split a time-indexed feature frame into train/val/test partitions.

    Supports two modes:
      1. Absolute dates: '2022-01-01' — used for initial training on fixed historical data
      2. Relative splits: 'last_Xpct' — used during retraining when data window grows
         e.g. val_split_date='last_20pct', test_split_date='last_10pct'
         means val = last 20% of data, test = last 10% of data (test is subset of val window)

    During normal initial training, absolute dates are used from params.yaml.
    During retraining triggered by drift, the retraining manager can pass relative
    splits so the model always trains on ~80% of available data regardless of
    how much new data has accumulated.
    """
    n = len(frame)

    # Parse val_split_date
    if isinstance(val_split_date, str) and val_split_date.startswith('last_') and val_split_date.endswith('pct'):
        val_pct = float(val_split_date.replace('last_', '').replace('pct', '')) / 100.0
        val_start_idx = int(n * (1 - val_pct))
        val_start_ts = frame.index[val_start_idx]
    else:
        val_start_ts = pd.Timestamp(val_split_date)

    # Parse test_split_date
    if isinstance(test_split_date, str) and test_split_date.startswith('last_') and test_split_date.endswith('pct'):
        test_pct = float(test_split_date.replace('last_', '').replace('pct', '')) / 100.0
        test_start_idx = int(n * (1 - test_pct))
        test_start_ts = frame.index[test_start_idx]
    else:
        test_start_ts = pd.Timestamp(test_split_date)

    # Validate that splits produce non-empty partitions
    train = frame.loc[frame.index < val_start_ts]
    val = frame.loc[(frame.index >= val_start_ts) & (frame.index < test_start_ts)]
    test = frame.loc[frame.index >= test_start_ts]

    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        if len(split_df) == 0:
            raise ValueError(
                f"Time split produced an empty '{split_name}' partition. "
                f"Data range: {frame.index.min().date()} to {frame.index.max().date()}. "
                f"val_split_date={val_split_date}, test_split_date={test_split_date}. "
                f"For retraining on new data windows, consider using relative splits like "
                f"val_split_date='last_20pct', test_split_date='last_10pct'."
            )

    feature_columns = [col for col in frame.columns if col != label_col]
    return (
        train[feature_columns], val[feature_columns], test[feature_columns],
        train[label_col], val[label_col], test[label_col],
    )


def _prepare_target_encoder(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray]:
    """Encode string labels into integer classes shared across all splits."""

    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_val, y_test], axis=0).astype(str))
    return encoder, encoder.transform(y_train.astype(str)), encoder.transform(y_val.astype(str)), encoder.transform(y_test.astype(str))


def _build_estimator(model_params: dict, y_train_enc=None, n_classes: int = 2) -> tuple:
    """
    Create the primary XGBoost classifier with automatic objective selection.

    For binary problems (2 classes) we use 'binary:logistic' which does not
    require num_class to be set. For multiclass problems we use
    'multi:softprob' with num_class set explicitly.

    sample_weight is computed using sklearn's compute_sample_weight so that
    minority classes (e.g. Bear at 2%) are not drowned out by the majority.
    """
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weight = None
    if y_train_enc is not None:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    if XGBClassifier is not None:
        if n_classes == 2:
            model = XGBClassifier(
                n_estimators=int(model_params["n_estimators"]),
                max_depth=int(model_params["max_depth"]),
                learning_rate=float(model_params["learning_rate"]),
                subsample=float(model_params["subsample"]),
                min_child_weight=int(model_params.get("min_child_weight", 1)),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=int(TRAINING_PARAMS["random_state"]),
                tree_method="hist",
            )
        else:
            model = XGBClassifier(
                n_estimators=int(model_params["n_estimators"]),
                max_depth=int(model_params["max_depth"]),
                learning_rate=float(model_params["learning_rate"]),
                subsample=float(model_params["subsample"]),
                min_child_weight=int(model_params.get("min_child_weight", 1)),
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=int(TRAINING_PARAMS["random_state"]),
                tree_method="hist",
            )
        return model, sample_weight

    logger.warning("xgboost unavailable; falling back to RandomForestClassifier.")
    model = RandomForestClassifier(
        n_estimators=int(model_params["n_estimators"]),
        max_depth=int(model_params["max_depth"]),
        class_weight="balanced",
        random_state=int(TRAINING_PARAMS["random_state"]),
    )
    return model, None


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
def _log_to_mlflow(
    run,
    regime_type: str,
    model: Any,
    scaler: StandardScaler,
    importance_path: Path,
    confusion_path: Path,
    model_name: str,
    feature_columns: list[str],
    metrics: dict[str, float],
    extra_params: dict[str, Any],
) -> tuple[str, Any]:
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

    try:
        # Works on newer MLflow servers that support logged-models.
        mlflow.sklearn.log_model(
            model,
            artifact_path=f"{regime_type}_model",
            registered_model_name=model_name,
        )
    except Exception as exc:
        msg = str(exc)
        if "/api/2.0/mlflow/logged-models" not in msg or "404" not in msg:
            raise

        logger.warning(
            "Logged-models endpoint is unavailable; using save_model + log_artifacts + create_model_version for %s",
            model_name,
        )

        import tempfile

        with tempfile.TemporaryDirectory(prefix=f"{regime_type}_mlflow_model_") as tmp_dir:
            local_model_dir = Path(tmp_dir) / "model"

            # Save model locally first, then log that directory into the run.
            mlflow.sklearn.save_model(model, path=str(local_model_dir))
            mlflow.log_artifacts(str(local_model_dir), artifact_path=f"{regime_type}_model")

            client = MlflowClient()

            # Ensure the registered model exists before creating a version.
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(model_name)

            mv = client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/{regime_type}_model",
                run_id=run_id,
            )
            model_version = mv.version

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
            versions = client.search_model_versions(f"name='{model_name}'")
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)
            if latest:
                client.set_registered_model_alias(model_name, "production", latest[0].version)
                logger.info("Model %s v%s -> alias 'production'", model_name, latest[0].version)
                model_version = latest[0].version
        except Exception as exc:
            logger.warning("Could not set model alias for %s: %s", model_name, exc)

    return run_id, model_version

# def _log_to_mlflow(run, regime_type: str, model: Any, scaler: StandardScaler, importance_path: Path, confusion_path: Path, model_name: str, feature_columns: list[str], metrics: dict[str, float], extra_params: dict[str, Any]) -> tuple[str, Any]:
#     """Log parameters, metrics, artifacts, and model registry entries to MLflow when available."""

#     run_id = run.info.run_id
#     model_version = None

#     if mlflow is None:
#         return run_id, model_version

#     mlflow.set_tag("regime_type", regime_type)
#     try:
#         git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
#     except Exception:
#         git_commit = "unknown"
#     mlflow.set_tag("git_commit", git_commit)
#     mlflow.log_params({**extra_params, "feature_count": len(feature_columns)})
#     mlflow.log_metrics(metrics)
#     mlflow.log_metric("train_test_f1_gap", abs(metrics.get("train_f1", 0.0) - metrics.get("test_f1", 0.0)))
#     mlflow.log_artifact(str(importance_path))
#     mlflow.log_artifact(str(confusion_path))
#     mlflow.sklearn.log_model(model, artifact_path=f"{regime_type}_model", registered_model_name=model_name)

#     scaler_path = MODELS_DIR / f"{regime_type}_scaler.pkl"
#     joblib.dump(scaler, scaler_path)
#     mlflow.log_artifact(str(scaler_path))

#     feature_path = MODELS_DIR / f"{regime_type}_feature_columns.json"
#     with feature_path.open("w", encoding="utf-8") as handle:
#         json.dump(feature_columns, handle, indent=2)
#     mlflow.log_artifact(str(feature_path))

#     if MlflowClient is not None:
#         try:
#             client = MlflowClient()
#             versions = client.search_model_versions(f"name='{model_name}'")
#             latest = sorted(versions, key=lambda v: int(v.version), reverse=True)
#             if latest:
#                 client.set_registered_model_alias(model_name, "production", latest[0].version)
#                 logger.info("Model %s v%s -> alias 'production'", model_name, latest[0].version)
#                 model_version = latest[0].version
#         except Exception as exc:
#             logger.warning("Could not set model alias for %s: %s", model_name, exc)

#     return run_id, model_version


def train_single_regime(regime_type: str, features_path: Path, label_col: str, experiment_name: str, model_name: str, model_params: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate a single regime model end to end."""

    frame = _load_feature_frame(features_path, label_col)
    logger.info("[%s] Loaded feature frame %s", regime_type, frame.shape)
    logger.info("[%s] Label distribution: %s", regime_type, frame[label_col].value_counts(dropna=False).to_dict())

    # Drop rows labelled 'Neutral' before splitting.
    # Neutral means the three labelling methods disagreed — it carries no
    # reliable signal and forces the model into a 3-class problem where
    # one class is meaningless noise. Dropping it restores the clean binary
    # classification that matched the notebook experiments.
    if "Neutral" in frame[label_col].values:
        before = len(frame)
        frame = frame[frame[label_col] != "Neutral"].copy()
        after = len(frame)
        logger.info(
            "[%s] Dropped %d Neutral rows -> %d rows remain for binary training",
            regime_type, before - after, after,
        )
    binary_dist = frame[label_col].value_counts()
    print(f"\n[{regime_type.upper()}] Binary label distribution after dropping Neutral:")
    for label_val, count in binary_dist.items():
        pct = 100 * count / len(frame)
        print(f"  {label_val}: {count} rows ({pct:.1f}%)")

    # Use env-var overrides for retraining runs; fall back to params for initial training
    val_split = os.getenv('RETRAIN_VAL_SPLIT', TRAINING_PARAMS['val_split_date'])
    test_split = os.getenv('RETRAIN_TEST_SPLIT', TRAINING_PARAMS['test_split_date'])

    x_train, x_val, x_test, y_train, y_val, y_test = _time_split(
        frame, label_col, val_split, test_split
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

    n_classes = len(encoder.classes_)
    xgb_model, sample_weight = _build_estimator(model_params, y_train_enc, n_classes=n_classes)
    xgb_model.fit(x_train_scaled, y_train_enc, sample_weight=sample_weight)

    lr_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(TRAINING_PARAMS["random_state"]))
    lr_model.fit(x_train_scaled, y_train_enc)

    xgb_train_metrics = _evaluate_split(xgb_model, x_train_scaled, y_train_enc)
    xgb_val_metrics = _evaluate_split(xgb_model, x_val_scaled, y_val_enc)
    xgb_test_metrics = _evaluate_split(xgb_model, x_test_scaled, y_test_enc)

    lr_val_metrics = _evaluate_split(lr_model, x_val_scaled, y_val_enc)
    lr_test_metrics = _evaluate_split(lr_model, x_test_scaled, y_test_enc)

    _format_comparison_table(
        regime_type,
        {"val_f1": xgb_val_metrics["f1"], "val_accuracy": xgb_val_metrics["accuracy"], "test_f1": xgb_test_metrics["f1"], "test_accuracy": xgb_test_metrics["accuracy"], "test_precision": xgb_test_metrics["precision"], "test_recall": xgb_test_metrics["recall"], "test_auc": xgb_test_metrics["auc"]},
        {"val_f1": lr_val_metrics["f1"], "val_accuracy": lr_val_metrics["accuracy"], "test_f1": lr_test_metrics["f1"], "test_accuracy": lr_test_metrics["accuracy"], "test_precision": lr_test_metrics["precision"], "test_recall": lr_test_metrics["recall"], "test_auc": lr_test_metrics["auc"]},
    )

    # Print class imbalance info so it's visible in logs and report
    from collections import Counter
    class_counts = Counter(y_train_enc)
    total = sum(class_counts.values())
    print(f"\n[{regime_type.upper()}] Training class distribution:")
    for cls_idx, count in sorted(class_counts.items()):
        cls_name = encoder.classes_[cls_idx]
        pct = 100 * count / total
        weight = total / (len(class_counts) * count)
        print(f"  {cls_name}: {count} samples ({pct:.1f}%) -> sample weight: {weight:.2f}x")

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