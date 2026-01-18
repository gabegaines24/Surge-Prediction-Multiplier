"""
Unit tests for retrieval.py data processing functions.
"""
import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add parent directory to path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import standardize_columns, create_time_series_features


class TestStandardizeColumns:
    """Tests for the standardize_columns function."""
    
    def test_standardize_yellow_taxi_columns(self, sample_yellow_df):
        """Test that yellow taxi columns are renamed correctly."""
        result = standardize_columns(sample_yellow_df, 'yellow')
        
        # Check that columns were renamed
        assert 'pickup_datetime' in result.columns
        assert 'dropoff_datetime' in result.columns
        
        # Original column names should be gone
        assert 'tpep_pickup_datetime' not in result.columns
        assert 'tpep_dropoff_datetime' not in result.columns
    
    def test_standardize_fhvhv_columns(self, sample_fhvhv_df):
        """Test that FHVHV columns remain properly named."""
        result = standardize_columns(sample_fhvhv_df, 'fhvhv')
        
        # FHVHV already has the standard names
        assert 'pickup_datetime' in result.columns
        assert 'dropoff_datetime' in result.columns
    
    def test_filters_to_required_columns_only(self, sample_yellow_df):
        """Test that only required columns are kept."""
        result = standardize_columns(sample_yellow_df, 'yellow')
        
        expected_columns = ['pickup_datetime', 'dropoff_datetime', 
                          'PULocationID', 'DOLocationID']
        
        assert list(result.columns) == expected_columns
        assert 'extra_column' not in result.columns
    
    def test_preserves_location_ids(self, sample_yellow_df):
        """Test that location IDs are preserved unchanged."""
        result = standardize_columns(sample_yellow_df, 'yellow')
        
        # Convert to list for comparison since we filtered columns
        original_pu = sample_yellow_df['PULocationID'].tolist()
        result_pu = result['PULocationID'].tolist()
        
        assert original_pu == result_pu


class TestCreateTimeSeriesFeatures:
    """Tests for the create_time_series_features function."""
    
    def test_creates_target_variable(self, sample_time_series_df):
        """Test that the target variable is created by shifting DER forward."""
        result = create_time_series_features(sample_time_series_df.copy())
        
        assert 'Target_DER_t+15' in result.columns
        
        # For zone 161, target at t=0 should equal DER at t=1
        zone_161 = result[result['Zone'] == 161].sort_values('Time_Bin')
        first_target = zone_161.iloc[0]['Target_DER_t+15']
        second_der = zone_161.iloc[1]['DER_t']
        
        assert abs(first_target - second_der) < 0.001
    
    def test_creates_lagged_features(self, sample_time_series_df):
        """Test that lagged DER features are created correctly."""
        result = create_time_series_features(sample_time_series_df.copy())
        
        assert 'Lag_DER_t-15' in result.columns
        assert 'Lag_DER_t-30' in result.columns
        
        # For zone 161, lag at t=1 should equal DER at t=0
        zone_161 = result[result['Zone'] == 161].sort_values('Time_Bin')
        second_lag = zone_161.iloc[1]['Lag_DER_t-15']
        first_der = zone_161.iloc[0]['DER_t']
        
        assert abs(second_lag - first_der) < 0.001
    
    def test_creates_demand_velocity(self, sample_time_series_df):
        """Test that demand velocity is calculated as difference in requests."""
        result = create_time_series_features(sample_time_series_df.copy())
        
        assert 'DemandVelocity_t' in result.columns
        assert 'Lag_DemandVelocity_t-15' in result.columns
        
        # For zone 161, velocity at t=1 should be ActiveRequests[1] - ActiveRequests[0]
        zone_161 = result[result['Zone'] == 161].sort_values('Time_Bin')
        velocity = zone_161.iloc[1]['DemandVelocity_t']
        expected = 120 - 100  # From fixture data
        
        assert abs(velocity - expected) < 0.001
    
    def test_operations_grouped_by_zone(self, sample_time_series_df):
        """Test that all operations respect zone boundaries."""
        result = create_time_series_features(sample_time_series_df.copy())
        
        # First row of each zone should have NaN for lagged features
        zone_161_first = result[result['Zone'] == 161].sort_values('Time_Bin').iloc[0]
        zone_237_first = result[result['Zone'] == 237].sort_values('Time_Bin').iloc[0]
        
        assert pd.isna(zone_161_first['Lag_DER_t-15'])
        assert pd.isna(zone_237_first['Lag_DER_t-15'])
        
        # Last row of each zone should have NaN for target (future value)
        zone_161_last = result[result['Zone'] == 161].sort_values('Time_Bin').iloc[-1]
        zone_237_last = result[result['Zone'] == 237].sort_values('Time_Bin').iloc[-1]
        
        assert pd.isna(zone_161_last['Target_DER_t+15'])
        assert pd.isna(zone_237_last['Target_DER_t+15'])
    
    def test_two_period_lag_correct(self, sample_time_series_df):
        """Test that 2-period lag (t-30) is correctly calculated."""
        result = create_time_series_features(sample_time_series_df.copy())
        
        # For zone 161, lag_t-30 at t=2 should equal DER at t=0
        zone_161 = result[result['Zone'] == 161].sort_values('Time_Bin')
        third_lag = zone_161.iloc[2]['Lag_DER_t-30']
        first_der = zone_161.iloc[0]['DER_t']
        
        assert abs(third_lag - first_der) < 0.001
