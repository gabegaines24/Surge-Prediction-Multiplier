# Dynamic Surge Prediction: Spatio-Temporal Modeling for Ride-Sharing

## Project Overview

This project builds a predictive engine designed to forecast ride-sharing Surge Multipliers (via Demand Excess Ratios) for specific geographic zones in New York City. By analyzing 70M+ rows of taxi trip data, the model predicts supply-demand imbalances 15 minutes into the future.

The goal is to transition from reactive surge pricing to proactive demand management, allowing platforms to signal drivers and adjust pricing before the imbalance occurs.

## Tech Stack

- **Language**: Python 3.x
- **Big Data Engine**: Dask (for out-of-core parallel processing of 70M+ records)
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: XGBoost / LightGBM
- **APIs**: Open-Meteo (Historical Weather), NYC TLC (Trip Records)
- **Environment**: Virtual Environments (.venv), Jupyter/Neovim

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

## Running Tests

This project includes comprehensive unit tests for all data processing functions.

### Install Test Dependencies

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage Report

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Test data retrieval functions
pytest tests/test_retrieval.py -v

# Test weather service
pytest tests/test_weather_service.py -v

# Test modeling functions
pytest tests/test_modeling.py -v
```

### Run a Specific Test

```bash
pytest tests/test_retrieval.py::TestStandardizeColumns::test_standardize_yellow_taxi_columns -v
```
