"""
Unit tests for weather_service.py.
"""
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather_service import fetch_nyc_weather


class TestFetchNYCWeather:
    """Tests for the fetch_nyc_weather function."""
    
    @patch('weather_service.openmeteo_requests.Client')
    @patch('weather_service.retry')
    @patch('weather_service.requests_cache.CachedSession')
    def test_api_called_with_correct_parameters(self, mock_cache, mock_retry, mock_client):
        """Test that the API is called with correct NYC coordinates and dates."""
        # Set up the mock chain
        mock_session = Mock()
        mock_cache.return_value = mock_session
        mock_retry.return_value = mock_session
        
        mock_openmeteo = Mock()
        mock_client.return_value = mock_openmeteo
        
        # Mock the response structure
        mock_response = Mock()
        mock_hourly = Mock()
        
        # Create mock data for 24 hours
        mock_hourly.Time.return_value = 1704067200  # 2024-01-01 00:00:00 UTC timestamp
        mock_hourly.TimeEnd.return_value = 1704153600  # 2024-01-02 00:00:00 UTC timestamp
        
        mock_var_0 = Mock()
        mock_var_0.ValuesAsNumpy.return_value = [10.0] * 24  # Temperature data
        mock_var_1 = Mock()
        mock_var_1.ValuesAsNumpy.return_value = [0.5] * 24  # Precipitation data
        
        mock_hourly.Variables.side_effect = [mock_var_0, mock_var_1]
        mock_response.Hourly.return_value = mock_hourly
        mock_openmeteo.weather_api.return_value = [mock_response]
        
        # Call the function
        result = fetch_nyc_weather("2025-01-01", "2025-01-02")
        
        # Verify API was called with NYC coordinates
        call_args = mock_openmeteo.weather_api.call_args
        params = call_args[0][1]  # Second positional argument is params dict
        
        assert params['latitude'] == 40.7128
        assert params['longitude'] == -74.0060
        assert params['start_date'] == "2025-01-01"
        assert params['end_date'] == "2025-01-02"
        assert 'temperature_2m' in params['hourly']
        assert 'precipitation' in params['hourly']
    
    @patch('weather_service.openmeteo_requests.Client')
    @patch('weather_service.retry')
    @patch('weather_service.requests_cache.CachedSession')
    def test_returns_dataframe_with_correct_columns(self, mock_cache, mock_retry, mock_client):
        """Test that returned DataFrame has expected structure."""
        # Set up mocks
        mock_session = Mock()
        mock_cache.return_value = mock_session
        mock_retry.return_value = mock_session
        
        mock_openmeteo = Mock()
        mock_client.return_value = mock_openmeteo
        
        mock_response = Mock()
        mock_hourly = Mock()
        
        # 24 hours of data
        mock_hourly.Time.return_value = 1704067200
        mock_hourly.TimeEnd.return_value = 1704153600
        
        mock_var_0 = Mock()
        mock_var_0.ValuesAsNumpy.return_value = [15.0] * 24
        mock_var_1 = Mock()
        mock_var_1.ValuesAsNumpy.return_value = [1.0] * 24
        
        mock_hourly.Variables.side_effect = [mock_var_0, mock_var_1]
        mock_response.Hourly.return_value = mock_hourly
        mock_openmeteo.weather_api.return_value = [mock_response]
        
        result = fetch_nyc_weather("2025-01-01", "2025-01-02")
        
        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'Time_Bin' in result.columns
        assert 'temp' in result.columns
        assert 'precip' in result.columns
        assert len(result.columns) == 3
    
    @patch('weather_service.openmeteo_requests.Client')
    @patch('weather_service.retry')
    @patch('weather_service.requests_cache.CachedSession')
    def test_resamples_to_15min_intervals(self, mock_cache, mock_retry, mock_client):
        """Test that hourly data is resampled to 15-minute intervals."""
        # Set up mocks
        mock_session = Mock()
        mock_cache.return_value = mock_session
        mock_retry.return_value = mock_session
        
        mock_openmeteo = Mock()
        mock_client.return_value = mock_openmeteo
        
        mock_response = Mock()
        mock_hourly = Mock()
        
        # 4 hours of data should become 16 entries (4 * 4 = 16 fifteen-minute intervals)
        mock_hourly.Time.return_value = 1704067200  # Start time
        mock_hourly.TimeEnd.return_value = 1704081600  # +4 hours
        
        mock_var_0 = Mock()
        mock_var_0.ValuesAsNumpy.return_value = [10.0, 11.0, 12.0, 13.0]
        mock_var_1 = Mock()
        mock_var_1.ValuesAsNumpy.return_value = [0.0, 0.5, 1.0, 0.5]
        
        mock_hourly.Variables.side_effect = [mock_var_0, mock_var_1]
        mock_response.Hourly.return_value = mock_hourly
        mock_openmeteo.weather_api.return_value = [mock_response]
        
        result = fetch_nyc_weather("2025-01-01", "2025-01-01")
        
        # After resampling 4 hourly values to 15min, we should have more rows
        # Each hour gets split into 4x 15-min intervals using forward fill
        assert len(result) > 4
        
        # Time bins should be 15 minutes apart
        time_diffs = result['Time_Bin'].diff().dropna()
        assert all(diff == pd.Timedelta('15min') for diff in time_diffs)
    
    @patch('weather_service.openmeteo_requests.Client')
    @patch('weather_service.retry')
    @patch('weather_service.requests_cache.CachedSession')
    def test_handles_date_range_correctly(self, mock_cache, mock_retry, mock_client):
        """Test that the correct date range is processed."""
        mock_session = Mock()
        mock_cache.return_value = mock_session
        mock_retry.return_value = mock_session
        
        mock_openmeteo = Mock()
        mock_client.return_value = mock_openmeteo
        
        mock_response = Mock()
        mock_hourly = Mock()
        
        # Single day of hourly data
        mock_hourly.Time.return_value = 1704067200
        mock_hourly.TimeEnd.return_value = 1704153600
        
        mock_var_0 = Mock()
        mock_var_0.ValuesAsNumpy.return_value = [10.0] * 24
        mock_var_1 = Mock()
        mock_var_1.ValuesAsNumpy.return_value = [0.0] * 24
        
        mock_hourly.Variables.side_effect = [mock_var_0, mock_var_1]
        mock_response.Hourly.return_value = mock_hourly
        mock_openmeteo.weather_api.return_value = [mock_response]
        
        result = fetch_nyc_weather("2025-01-15", "2025-01-16")
        
        # Verify the dates were passed correctly
        call_args = mock_openmeteo.weather_api.call_args[0][1]
        assert call_args['start_date'] == "2025-01-15"
        assert call_args['end_date'] == "2025-01-16"
