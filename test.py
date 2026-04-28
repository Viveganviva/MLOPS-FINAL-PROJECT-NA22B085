from src.predict import fetch_recent_data, _build_features_for_regime

data = fetch_recent_data("SPY", regime_type="trend")

for k, df in data.items():
    print("\n======", k, "======")
    print(df.columns)

feat = _build_features_for_regime("trend", data, "SPY")

print("\n====== FEATURE COLUMNS ======")
print(feat.columns.tolist()[:20])
print("Has Close?", "Close" in feat.columns)