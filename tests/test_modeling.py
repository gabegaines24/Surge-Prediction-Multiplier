"""
Unit tests for modeling.py functions.
"""
import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling import prepare_cyclical_features, train_surge_model


class TestPrepareCyclicalFeatures:
    """Tests for the prepare_cyclical_features function."""
    
    def test_extracts_hour_from_time_bin(self):
        """Test that hour is correctly extracted from Time_Bin."""
        df = pd.DataFrame({
            'Time_Bin': pd.to_datetime(['2025-01-01 00:00:00', 
                                       '2025-01-01 14:30:00',
                                       '2025-01-01 23:45:00'])
        })
        
        result = prepare_cyclical_features(df)
        
        assert 'hour' in result.columns
        assert result.iloc[0]['hour'] == 0
        assert result.iloc[1]['hour'] == 14
        assert result.iloc[2]['hour'] == 23
    
    def test_creates_sine_cosine_features(self):
        """Test that sine and cosine transformations are created."""
        df = pd.DataFrame({
            'Time_Bin': pd.to_datetime(['2025-01-01 00:00:00', 
                                       '2025-01-01 06:00:00',
                                       '2025-01-01 12:00:00',
                                       '2025-01-01 18:00:00'])
        })
        
        result = prepare_cyclical_features(df)
        
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
    
    def test_midnight_wraps_correctly(self):
        """Test that midnight (00:00) and 23:00 are close in cyclical space."""
        df = pd.DataFrame({
            'Time_Bin': pd.to_datetime(['2025-01-01 00:00:00',  # Midnight
                                       '2025-01-01 23:00:00'])  # 11 PM
        })
        
        result = prepare_cyclical_features(df)
        
        # At midnight (hour=0), sin should be ~0 and cos should be ~1
        midnight_sin = result.iloc[0]['hour_sin']
        midnight_cos = result.iloc[0]['hour_cos']
        
        assert abs(midnight_sin - 0.0) < 0.01
        assert abs(midnight_cos - 1.0) < 0.01
        
        # At 23:00, we're close to completing the cycle, so values should be similar
        # The Euclidean distance in (sin, cos) space should be small
        evening_sin = result.iloc[1]['hour_sin']
        evening_cos = result.iloc[1]['hour_cos']
        
        distance = np.sqrt((midnight_sin - evening_sin)**2 + (midnight_cos - evening_cos)**2)
        # They should be relatively close (within ~0.5 in the unit circle)
        assert distance < 0.5
    
    def test_noon_values_correct(self):
        """Test specific values at noon (12:00)."""
        df = pd.DataFrame({
            'Time_Bin': pd.to_datetime(['2025-01-01 12:00:00'])
        })
        
        result = prepare_cyclical_features(df)
        
        # At hour=12, we're halfway through the cycle
        # sin(2π * 12/24) = sin(π) ≈ 0
        # cos(2π * 12/24) = cos(π) ≈ -1
        noon_sin = result.iloc[0]['hour_sin']
        noon_cos = result.iloc[0]['hour_cos']
        
        assert abs(noon_sin - 0.0) < 0.01
        assert abs(noon_cos - (-1.0)) < 0.01
    
    def test_original_dataframe_columns_preserved(self):
        """Test that original columns are not removed."""
        df = pd.DataFrame({
            'Time_Bin': pd.to_datetime(['2025-01-01 00:00:00', 
                                       '2025-01-01 12:00:00']),
            'other_column': [1, 2]
        })
        
        result = prepare_cyclical_features(df)
        
        # Original columns should still be there
        assert 'Time_Bin' in result.columns
        assert 'other_column' in result.columns
        
        # New columns added
        assert 'hour' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns


class TestTrainSurgeModel:
    """Tests for the train_surge_model function."""
    
    def test_returns_trained_model(self, sample_training_data):
        """Test that function returns a trained XGBoost model."""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = train_surge_model(X_train, y_train, X_test, y_test)
        
        # Should return an XGBoost model object
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
    
    def test_model_can_make_predictions(self, sample_training_data):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = train_surge_model(X_train, y_train, X_test, y_test)
        predictions = model.predict(X_test)
        
        # Should return predictions with correct shape
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_model_learns_patterns(self, sample_training_data):
        """Test that model produces reasonable predictions."""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = train_surge_model(X_train, y_train, X_test, y_test)
        predictions = model.predict(X_test)
        
        # Predictions should be in a reasonable range
        # Our synthetic data has y values roughly between 0 and 3
        assert all(p > -5 for p in predictions)  # Not absurdly negative
        assert all(p < 10 for p in predictions)  # Not absurdly positive
    
    def test_handles_feature_names(self, sample_training_data):
        """Test that model properly handles feature columns."""
        X_train, X_test, y_train, y_test = sample_training_data
        
        model = train_surge_model(X_train, y_train, X_test, y_test)
        
        # Model should work with the same features
        predictions = model.predict(X_test[X_train.columns])
        assert len(predictions) == len(X_test)
