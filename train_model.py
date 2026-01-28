"""
Train and save XGBoost model for deployment
This creates a .pkl file that api.py can load
"""

import pandas as pd
import numpy as np
import joblib
import os
from modeling import prepare_cyclical_features, train_surge_model
from weather_service import fetch_nyc_weather

# Paths
PROCESSED_DIR = './processed_data/'
MODEL_DIR = './models/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Features and target (using actual column names from retrieval.py)
FEATURES = ['SupplyElasticity', 'Lag_DER_t-15', 'Lag_DER_t-30', 
            'DemandVelocity_t', 'Lag_DemandVelocity_t-15']
TARGET = 'Target_DER_t+15'

def train_and_save():
    """Train model and save for API deployment"""
    
    print("=" * 60)
    print("TRAINING MODEL FOR DEPLOYMENT")
    print("=" * 60)
    
    # 1. Load processed data
    print("\n[1/5] Loading processed taxi data...")
    df_train = pd.read_parquet(f'{PROCESSED_DIR}/train_data.parquet')
    df_test = pd.read_parquet(f'{PROCESSED_DIR}/test_data.parquet')
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}")
    
    # 2. Add weather
    print("\n[2/5] Fetching weather data...")
    df_weather = fetch_nyc_weather("2025-01-01", "2025-03-31")
    df_train = pd.merge(df_train, df_weather, on='Time_Bin', how='left')
    df_test = pd.merge(df_test, df_weather, on='Time_Bin', how='left')
    
    # 3. Add cyclical features
    print("\n[3/5] Engineering features...")
    df_train = prepare_cyclical_features(df_train)
    df_test = prepare_cyclical_features(df_test)
    
    # Add day of week (0=Monday, 6=Sunday)
    df_train['day_of_week'] = df_train['Time_Bin'].dt.dayofweek
    df_test['day_of_week'] = df_test['Time_Bin'].dt.dayofweek
    
    # Add weekend flag (Fridays surge more!)
    df_train['is_weekend'] = (df_train['day_of_week'] >= 5).astype(int)
    df_test['is_weekend'] = (df_test['day_of_week'] >= 5).astype(int)
    
    # 4. Prepare final features
    all_features = FEATURES + ['temp', 'precip', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend']
    df_train = df_train.dropna(subset=all_features + [TARGET])
    df_test = df_test.dropna(subset=all_features + [TARGET])
    
    X_train = df_train[all_features]
    y_train = df_train[TARGET]
    X_test = df_test[all_features]
    y_test = df_test[TARGET]
    
    print(f"  Final shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # 5. Train model
    print("\n[4/5] Training XGBoost model...")
    model = train_surge_model(X_train, y_train, X_test, y_test)
    
    # 6. Save model
    print("\n[5/5] Saving model...")
    model_path = f'{MODEL_DIR}/xgboost_surge_model.pkl'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to: {model_path}")
    
    # Save feature names for reference
    feature_info = {
        'features': all_features,
        'target': TARGET,
        'model_type': 'XGBoost',
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    joblib.dump(feature_info, f'{MODEL_DIR}/model_info.pkl')
    
    print("\n" + "=" * 60)
    print("✓ MODEL READY FOR DEPLOYMENT")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start API: python api.py")
    print("  2. Test endpoint: curl http://localhost:5000/health")
    print("  3. Start frontend: cd frontend && npm run dev")
    print("=" * 60)
    
    return model

if __name__ == '__main__':
    train_and_save()
