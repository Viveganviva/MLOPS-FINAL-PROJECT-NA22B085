# User Manual

## What This App Does

This app looks at market data and tells you whether the market is trending, quiet or volatile, and whether the broader market is behaving like a bull or a bear. It is meant for fast checking and monitoring, not for fully automated trading.

## How to Start It

Run:

```bash
docker-compose up
```

If you want all services built from scratch in the background, use:

```bash
docker-compose up --build -d
```

## How to Use the Prediction Screen

1. Open `http://localhost:80`.
2. Choose a ticker from the drop-down or type a custom ticker.
3. Optionally choose a date.
4. Click `Analyze`.
5. Read the three cards:
	- Trend regime
	- Volatility regime
	- Market direction
6. Check the confidence bar under each card.
7. Look at the history chart to see recent prediction activity.

## What the Results Mean

- `Trending` means price movement is persistent in one direction.
- `MeanReverting` means price is moving back and forth around a fair value.
- `HighVol` means the market is moving more than usual and risk is higher.
- `LowVol` means the market is calmer and position sizes can usually be smaller or more stable.
- `Bull` means the broad market is showing positive strength.
- `Bear` means the broad market is weak and defensive behavior may make more sense.

## How to Read Confidence

Confidence is a number from 0 to 1. Higher means the model is more certain about the result. A value close to 1 is stronger than a value close to 0.5. High confidence does not guarantee the label is correct, but low confidence means the signal is weaker.

## Frequently Asked Questions

### Do I need to run the Python scripts manually?

Not if you are using Docker after the models are trained. The usual flow is: ingest data, build features, train models, then start Docker.

### Why is SPY the default?

SPY is the main reference ticker for the project and works well as a baseline for equity regime analysis.

### Where are the predictions saved?

They are appended to `data/simulation/prediction_log.csv`.

### Where are the trained models saved?

They are saved in `models/` and also tracked in MLflow when that service is running.

### What if the API says the service is degraded?

That usually means the app is running, but one or more supporting services or models are not fully ready yet.

## Troubleshooting

### API not responding

- Check that `docker-compose up` is running.
- Open `http://localhost:8000/health`.
- If the port is busy, stop any older Docker stack first.

### No models loaded

- Run `python src/data_ingestion.py`.
- Run `python src/feature_engineering.py`.
- Run `python src/train.py --force`.
- Then restart Docker.

### Frontend loads but prediction fails

- Make sure the API container is healthy.
- Check whether the model files exist in `models/`.
- Confirm that the machine has internet access for `yfinance` if you are running predictions outside the Docker image.

### MLflow page is empty

- Make sure the training step completed successfully.
- Confirm that `mlruns/` exists and Docker can mount it.

### Grafana shows no data

- Check that Prometheus can reach the API on port 8001.
- Wait a minute after startup so metrics have time to appear.

## Practical Advice

- Use `Trending` and `Bull` as supportive signals, not automatic trade triggers.
- Treat `HighVol` as a warning that position sizing and stop-loss discipline matter more.
- If confidence is low and drift is high, trust the pipeline screen more than the prediction screen.