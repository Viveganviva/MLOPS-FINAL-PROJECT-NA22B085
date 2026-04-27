# Low-Level Design

The implementation is divided into small modules so each pipeline stage can be tested independently.

- `src/data_ingestion.py` handles data acquisition and raw storage.
- `src/feature_engineering.py` derives indicators and regime features.
- `src/train.py` orchestrates model fitting and artifact persistence.
- `src/predict.py` provides a narrow prediction interface.
- `src/drift_monitor.py` compares live and reference distributions.
- `src/retraining_manager.py` applies the retraining policy.