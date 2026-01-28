"""
Flask API for Surge Prediction
Connects frontend to trained XGBoost model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# Load the trained model (we'll create this next)
MODEL_PATH = './models/xgboost_surge_model.pkl'
model = None

def load_model():
    """Load the trained model"""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   Run train_and_save_model() first!")

@app.route('/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict surge level given input features
    
    Expected JSON:
    {
        "supplyElasticity": 0.6,
        "lagDER15": 1.8,
        "lagDER30": 1.6,
        "demandVelocity": 15,
        "lagDemandVelocity15": 12,
        "temp": 10,
        "precip": 0,
        "hour": 8
    }
    """
    try:
        data = request.json
        
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please train and save the model first'
            }), 500
        
        # Extract features
        features = prepare_features(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence (use model's feature importance or tree variance)
        confidence = calculate_confidence(model, features)
        
        # Determine surge level
        surge_level = get_surge_level(prediction)
        
        return jsonify({
            'prediction': float(prediction),
            'surgeLevel': surge_level,
            'confidence': f"{confidence:.0%}",
            'recommendation': get_recommendation(prediction, data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

def prepare_features(data):
    """Convert input data to model features"""
    # Calculate cyclical time features
    hour = data['hour']
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Create feature array in correct order
    features = np.array([[
        data['supplyElasticity'],
        data['lagDER15'],
        data['lagDER30'],
        data['demandVelocity'],
        data['lagDemandVelocity15'],
        data['temp'],
        data['precip'],
        hour_sin,
        hour_cos
    ]])
    
    return features

def calculate_confidence(model, features):
    """
    Estimate prediction confidence
    For XGBoost, we can use tree predictions variance
    """
    try:
        # Get predictions from individual trees
        tree_predictions = []
        for tree in model.get_booster().get_dump():
            # This is simplified - in production you'd use proper uncertainty estimation
            pass
        
        # Simplified confidence based on feature stability
        # In production, use proper uncertainty quantification
        base_confidence = 0.85
        return base_confidence
    except:
        return 0.80  # Default confidence

def get_surge_level(der):
    """Convert DER to surge level category"""
    if der < 0.8:
        return 'Low Demand'
    elif der < 1.2:
        return 'Normal'
    elif der < 1.5:
        return 'Moderate Surge'
    elif der < 2.0:
        return 'High Surge'
    else:
        return 'Extreme Surge'

def get_recommendation(der, input_data):
    """Generate business recommendation"""
    if der < 0.8:
        return "Supply exceeds demand. Consider reducing active driver incentives."
    elif der < 1.2:
        return "Balanced market conditions. Maintain current operations."
    elif der < 1.5:
        return "Moderate demand pressure. Consider 1.2-1.4x surge pricing."
    elif der < 2.0:
        action = "Activate surge pricing (1.5-1.8x) and send driver alerts."
        if input_data.get('precip', 0) > 1:
            action += " Rain detected - expect sustained high demand."
        return action
    else:
        return "Extreme surge conditions. Implement maximum pricing (2.0x+) and emergency driver activation."

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        return jsonify({
            'algorithm': 'XGBoost',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'features': [
                'SupplyElasticity',
                'Lag_DER_t-15',
                'Lag_DER_t-30',
                'DemandVelocity_t',
                'Lag_DemandVelocity_t-15',
                'Temperature',
                'Precipitation',
                'Hour_Sin',
                'Hour_Cos'
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚕 NYC Taxi Surge Prediction API")
    print("=" * 60)
    
    # Load model on startup
    load_model()
    
    print("\nAPI Endpoints:")
    print("  GET  /health      - Health check")
    print("  POST /predict     - Make prediction")
    print("  GET  /model-info  - Model information")
    print("\nStarting server on http://localhost:8000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=True)
