# Dynamic Surge Prediction: Spatio-Temporal Modeling for Ride-Sharing

## Project Overview

This project builds a predictive engine designed to forecast ride-sharing Surge Multipliers (via Demand Excess Ratios) for specific geographic zones in New York City. By analyzing 70M+ rows of taxi trip data, the model predicts supply-demand imbalances 15 minutes into the future.

The goal is to transition from reactive surge pricing to proactive demand management, allowing platforms to signal drivers and adjust pricing before the imbalance occurs.

## Tech Stack

- **Language**: Python 3.x
- **Big Data Engine**: Dask (for out-of-core parallel processing of 70M+ records)
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: XGBoost (with optional hyperparameter tuning & time-series CV)
- **API**: FastAPI + Uvicorn (loads feature order from `models/model_info.pkl`)
- **Config**: `backend/config.yaml` (paths, model hyperparameters, MLflow toggle, outlier bounds)
- **APIs**: Open-Meteo (Historical Weather), NYC TLC (Trip Records)
- **Environment**: Virtual Environments (.venv), Jupyter/Neovim

## Repo Layout

1. `frontend/`: React UI
2. `backend/`: FastAPI server + training/retrieval pipeline code
3. `models/`: saved model artifacts (`xgboost_surge_model*.pkl`, `model_info.pkl`)
4. `processed_data/`, `taxi data/`: intermediate and raw data (Dask pipeline)

## Data Pipeline & Architecture

### 1. Data Ingestion

Raw trip data is ingested from NYC TLC Parquet files. Due to the massive volume (70M+ rows), we utilize Dask to perform lazy loading and parallel aggregation, preventing memory (RAM) overflow.

### 2. Spatio-Temporal Binning

The city is divided into TLC Taxi Zones, and time is discretized into 15-minute intervals.

- **Demand**: Count of pickups per [Zone, Time] bin
- **Supply**: Count of dropoffs per [Zone, Time] bin (proxy for driver availability)

### 3. Feature Engineering

The model utilizes three categories of features:

- **Core Spatio-Temporal**: Supply Elasticity and Demand Velocity
- **Lagged Features**: Historical demand and surge values from t-15 and t-30 minutes to capture autocorrelation
- **Exogenous Shocks**: Real-time weather data (precipitation, temperature) and scheduled event data (stadium start/end times)

## Modeling Approach

### Target Variable (y)

The Demand Excess Ratio (DER) at t+15 minutes:

```
y = Active_Requests(t+15) / Available_Drivers(t+15)
```

### Validation Strategy: Walk-Forward Validation

Standard random splits are avoided to prevent data leakage. We implement a strict time-series split:

- **Training**: January – October
- **Testing**: November (Unseen "future" data)

## Training & serving

1. **Process data (Dask)**: `python -m backend.retrieval`
2. **Train & save model**: `python -m backend.train_model` (writes `models/xgboost_surge_model.pkl`, versioned copy, and `model_info.pkl`)
3. **Run API**: `uvicorn backend.api:app --host 0.0.0.0 --port 8000`

Optional: set `mlflow.enabled: true` in `backend/config.yaml` or `SURGE_MLFLOW_ENABLED=1` for experiment tracking.

## Running Tests

This project includes comprehensive unit tests for all data processing functions.

### Install Test Dependencies

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest backend/tests/
```

### Run with Coverage Report

```bash
pytest backend/tests/ --cov=. --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Test data retrieval functions
pytest backend/tests/test_retrieval.py -v

# Test weather service
pytest backend/tests/test_weather_service.py -v

# Test modeling functions
pytest backend/tests/test_modeling.py -v
```

### Run a Specific Test

```bash
pytest backend/tests/test_retrieval.py::TestStandardizeColumns::test_standardize_yellow_taxi_columns -v
```
