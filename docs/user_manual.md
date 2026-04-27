# User Manual

## Setup

1. Run the scaffold script.
2. Populate `data/raw` with market history.
3. Build features and train models.
4. Start the API and the frontend.

## Operational Notes

- Prediction logs are written to `data/simulation/prediction_log.csv`.
- Pipeline parameters are centralized in `params.yaml`.
- Drift thresholds and retraining policy are also parameterized.