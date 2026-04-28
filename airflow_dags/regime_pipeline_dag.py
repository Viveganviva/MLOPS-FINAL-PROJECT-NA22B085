# """Airflow DAG scaffold for the regime detection pipeline."""

# from __future__ import annotations

# from datetime import datetime

# try:  # pragma: no cover - optional dependency guard for scaffold imports
#     from airflow import DAG
#     from airflow.operators.bash import BashOperator
# except ImportError:  # pragma: no cover - handled gracefully for the scaffold
#     DAG = None  # type: ignore[assignment]
#     BashOperator = None  # type: ignore[assignment]


# def build_dag() -> object | None:
#     if DAG is None or BashOperator is None:
#         return None

#     with DAG(
#         dag_id="regime_pipeline",
#         start_date=datetime(2024, 1, 1),
#         schedule=None,
#         catchup=False,
#         tags=["market-regime", "mlops"],
#     ) as dag:
#         BashOperator(task_id="ingest_data", bash_command="python src/data_ingestion.py")
#         BashOperator(task_id="featurize", bash_command="python src/feature_engineering.py")
#         BashOperator(task_id="train_models", bash_command="python src/train.py")

#     return dag


# dag = build_dag()


from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {"owner": "regime", "retries": 0}

with DAG(
    dag_id="regime_pipeline",
    default_args=default_args,
    description="Daily regime pipeline orchestration",
    schedule_interval="0 18 * * 1-5",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "regime"],
) as dag:

    health_check = BashOperator(
        task_id="health_check",
        bash_command="curl -f http://api:8000/health || exit 1",
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command="curl -X GET http://api:8000/drift/trend",
    )

    trigger_retrain = BashOperator(
        task_id="trigger_retrain_if_needed",
        bash_command="curl -X POST 'http://api:8000/retrain?triggered_by=airflow_schedule'",
    )

    health_check >> drift_check >> trigger_retrain