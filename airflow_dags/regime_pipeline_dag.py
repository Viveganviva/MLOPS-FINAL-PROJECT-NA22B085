"""Airflow DAG scaffold for the regime detection pipeline."""

from __future__ import annotations

from datetime import datetime

try:  # pragma: no cover - optional dependency guard for scaffold imports
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover - handled gracefully for the scaffold
    DAG = None  # type: ignore[assignment]
    BashOperator = None  # type: ignore[assignment]


def build_dag() -> object | None:
    if DAG is None or BashOperator is None:
        return None

    with DAG(
        dag_id="regime_pipeline",
        start_date=datetime(2024, 1, 1),
        schedule=None,
        catchup=False,
        tags=["market-regime", "mlops"],
    ) as dag:
        BashOperator(task_id="ingest_data", bash_command="python src/data_ingestion.py")
        BashOperator(task_id="featurize", bash_command="python src/feature_engineering.py")
        BashOperator(task_id="train_models", bash_command="python src/train.py")

    return dag


dag = build_dag()