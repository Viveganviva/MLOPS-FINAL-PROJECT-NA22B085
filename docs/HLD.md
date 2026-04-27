# High-Level Design

This project is organized as a repeatable MLOps system:

- ingest historical market data
- engineer trend, volatility, and structural features
- train regime classifiers
- serve predictions through an API
- simulate live inference and monitor drift
- trigger retraining when the operating envelope changes