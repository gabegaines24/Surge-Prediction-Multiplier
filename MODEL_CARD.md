# Model Card: NYC TLC Demand Excess Ratio (DER) Forecaster

This document follows common **Model Card** practice (Mitchell et al., 2019) for transparency and responsible use. It applies to the XGBoost regression model served by `backend/api.py` and trained via `python -m backend.train_model`.

## Model details

| Field | Description |
|--------|-------------|
| **Organization / type** | Independent research / educational demo; not a commercial ride-share product. |
| **Versioning** | Artifacts written to `models/xgboost_surge_model.pkl` with timestamped copies; metadata in `models/model_info.pkl`. |
| **Algorithm** | XGBoost regressor (`reg:squarederror`), with optional early stopping on a time-ordered validation split during training. |
| **Target** | `Target_DER_t+15`: Demand Excess Ratio (DER) at 15 minutes ahead, derived from TLC trip aggregates in 15-minute bins per zone. |
| **Inputs** | Engineered features including lagged DER, demand velocity, weather (Open-Meteo), calendar/rush-hour flags, and TLC zone hints (e.g. airport / Manhattan core). Exact columns are listed in `model_info.pkl` after training. |

## Intended use

- **Primary:** Explore relationships between market dynamics, weather, time, and zone context and a **scalar DER forecast** for discussion and prototyping.
- **Out of scope:** Automated surge pricing, driver payouts, compliance, or real-time dispatch without a full production MLOps stack, monitoring, and legal review.

## Metrics

Training logs report **MAE, RMSE, R², MAPE** on the held-out period. Time-series CV may report **cv_mae_mean** / **cv_mae_std**. These are **historical** on the specific train/test split documented in `backend/config.yaml` and echoed in `GET /model-info` under `data` after retraining.

## Data and split

- **Sources:** NYC TLC yellow / FHVHV Parquet under `paths.taxi_data_dir` (see `backend/retrieval.py`), merged with historical weather for `weather_start`–`weather_end`.
- **Train/test rule:** Rows with `Time_Bin < train_test_split_date` → train; `Time_Bin >= train_test_split_date` → test. This is a **time-based** split to reduce leakage; it does not guarantee stationarity or representativeness of future markets.

## Limitations

1. **Supply proxy:** “Available drivers” is approximated from dropoff counts, not live fleet telemetry.
2. **Weather:** Gridded historical/Open-Meteo values may not match street-level conditions.
3. **Generalization:** The model is fit on NYC TLC-style aggregates; other cities, regulations, or products are out-of-distribution.
4. **Lag features:** API defaults may substitute rolling statistics; see `defaultsUsed` in responses.
5. **Extreme DER:** Rows can be clipped by `der_outlier_upper` / `der_outlier_lower` in config; tail risk may be underrepresented.

## Uncertainty and interpretability (API)

| Mechanism | Meaning |
|-----------|---------|
| **Tree ensemble dispersion** (`uncertainty.treeEnsembleDispersion`) | Heuristic from spread of per-tree predictions; **not** a calibrated probability. |
| **Holdout residual band** (`uncertainty.holdoutErrorBand`) | When present, uses empirical p10–p90 of **test-set residuals**; useful as a **rough** error range, **not** a formal prediction interval. |
| **GET `/interpretability/global-importance`** | XGBoost **gain** importances (correlative, not causal). |
| **POST `/interpretability/shap`** | **TreeSHAP** values for a single input row (local explanation). |

Re-training updates calibration statistics in `model_info.pkl`.

## Known biases and fairness considerations

- **Geographic:** TLC zones and airport/Manhattan flags encode structural demand patterns; underserved areas may be underrepresented if trip volume is low.
- **Temporal:** Training windows favor specific seasons and events; holidays and anomalies may be sparse.
- **Modal / fleet:** Yellow/FHVHV trips do not match every TNC’s rider mix or driver incentives.

Downstream use for **pricing or access** decisions should include fairness review beyond this card.

## Environmental impact

Training uses CPU-heavy Dask aggregation and XGBoost; energy use scales with data volume and tuning. No GPU requirement is assumed for the default pipeline.

## How to report issues

Open an issue on the project repository or document limitations when citing this demo in portfolios or coursework.

## References

- Mitchell, M. et al., “Model Cards for Model Reporting” (2019).
- NYC TLC data: [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Open-Meteo: [open-meteo.com](https://open-meteo.com/).
