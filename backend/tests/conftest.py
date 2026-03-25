"""
Shared test fixtures for unit tests.
"""
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_yellow_df():
    """Create a sample yellow taxi DataFrame for testing."""
    return pd.DataFrame({
        'tpep_pickup_datetime': pd.date_range('2025-01-01', periods=5, freq='15min'),
        'tpep_dropoff_datetime': pd.date_range('2025-01-01 00:10:00', periods=5, freq='15min'),
        'PULocationID': [161, 161, 237, 237, 161],
        'DOLocationID': [162, 163, 238, 239, 162],
        'extra_column': [1, 2, 3, 4, 5]  # Should be filtered out
    })


@pytest.fixture
def sample_fhvhv_df():
    """Create a sample FHVHV (Uber/Lyft) DataFrame for testing."""
    return pd.DataFrame({
        'pickup_datetime': pd.date_range('2025-01-01', periods=5, freq='15min'),
        'dropoff_datetime': pd.date_range('2025-01-01 00:10:00', periods=5, freq='15min'),
        'PULocationID': [161, 161, 237, 237, 161],
        'DOLocationID': [162, 163, 238, 239, 162],
        'extra_column': [1, 2, 3, 4, 5]  # Should be filtered out
    })


@pytest.fixture
def sample_time_series_df():
    """Create sample time-series data with multiple zones."""
    # Create data for 2 zones across 6 time periods
    time_bins = pd.date_range('2025-01-01', periods=6, freq='15min')
    
    # Zone 161 has increasing demand
    zone_161_data = {
        'Zone': [161] * 6,
        'Time_Bin': time_bins,
        'DER_t': [1.0, 1.2, 1.5, 1.8, 2.0, 2.2],
        'ActiveRequests': [100, 120, 150, 180, 200, 220],
        'SupplyElasticity': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    }
    
    # Zone 237 has varying demand
    zone_237_data = {
        'Zone': [237] * 6,
        'Time_Bin': time_bins,
        'DER_t': [1.5, 1.4, 1.6, 1.7, 1.5, 1.6],
        'ActiveRequests': [150, 140, 160, 170, 150, 160],
        'SupplyElasticity': [1.0, 0.95, 1.05, 1.1, 1.0, 1.05]
    }
    
    df_161 = pd.DataFrame(zone_161_data)
    df_237 = pd.DataFrame(zone_237_data)
    
    return pd.concat([df_161, df_237], ignore_index=True)


@pytest.fixture
def sample_training_data():
    """Create small training and test datasets."""
    np.random.seed(42)
    n_samples = 50
    
    # Generate synthetic features
    X = pd.DataFrame({
        'SupplyElasticity': np.random.uniform(0.5, 2.0, n_samples),
        'Lag_DER_t-15': np.random.uniform(0.8, 2.5, n_samples),
        'Lag_DER_t-30': np.random.uniform(0.8, 2.5, n_samples),
        'DemandVelocity_t': np.random.uniform(-50, 50, n_samples),
        'Lag_DemandVelocity_t-15': np.random.uniform(-50, 50, n_samples),
        'temp': np.random.uniform(-5, 25, n_samples),
        'precip': np.random.uniform(0, 5, n_samples),
        'hour_sin': np.random.uniform(-1, 1, n_samples),
        'hour_cos': np.random.uniform(-1, 1, n_samples)
    })
    
    # Generate target variable with some correlation to features
    y = (X['SupplyElasticity'] * 0.5 + 
         X['Lag_DER_t-15'] * 0.3 + 
         np.random.normal(0, 0.2, n_samples))
    
    # Split into train and test
    split_idx = int(n_samples * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test
