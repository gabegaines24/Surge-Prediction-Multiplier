# NYC Taxi Surge Predictor - Frontend

A clean, modern web interface for predicting taxi surge pricing using machine learning.

## Features

- 🎨 **Clean UI**: Modern, responsive design
- 📊 **Real-time Predictions**: Instant surge level forecasting
- 🌧️ **Weather Integration**: Incorporates weather data
- 📈 **Sample Scenarios**: Pre-loaded rush hour and weather scenarios
- 🎯 **Visual Feedback**: Color-coded surge levels

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development
- **Modern CSS** with CSS Variables

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Usage

1. **Manual Input**: Enter feature values manually
2. **Sample Scenarios**: Click pre-loaded scenarios (Morning Rush, Evening Rush, Rainy Day)
3. **Predict**: Click "Predict Surge" to get forecast
4. **Review Results**: See surge level, confidence, and recommendations

## Features Explained

### Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Supply Elasticity | Ratio of available drivers to active requests | 0-2 |
| DER (t-15/t-30) | Demand Excess Ratio from 15/30 minutes ago | 0.5-3 |
| Demand Velocity | Rate of change in trip requests | -50 to 50 |
| Temperature | Current temperature in Celsius | -20 to 40 |
| Precipitation | Rainfall in mm | 0-20 |
| Hour | Hour of day (24-hour format) | 0-23 |

### Surge Levels

- **Low Demand** (< 0.8): More drivers than riders
- **Normal** (0.8-1.2): Balanced supply/demand
- **Moderate Surge** (1.2-1.5): Demand exceeding supply
- **High Surge** (1.5-2.0): Significant demand pressure
- **Extreme Surge** (2.0+): Very high demand, limited supply

## Connecting to Backend

To connect this frontend to your Python backend:

1. Start the FastAPI backend (from project root, after `python train_model.py`):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API loads the trained model and feature list from `models/`. `POST /predict` accepts the same JSON fields as this frontend (`supplyElasticity`, `lagDER15`, `dayOfWeek`, etc.).

2. Ensure `App.tsx` calls the API (default port **8000**):

```typescript
const handlePredict = async () => {
  setLoading(true)
  
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input)
    })
    
    const data = await response.json()
    // Process response...
  } catch (error) {
    console.error('Prediction failed:', error)
  }
  
  setLoading(false)
}
```

## License

MIT
