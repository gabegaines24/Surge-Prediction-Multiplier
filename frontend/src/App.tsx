import { useEffect, useState } from 'react'
import './App.css'

interface PredictionInput {
  supplyElasticity: number
  lagDER15: number
  lagDER30: number
  demandVelocity: number
  lagDemandVelocity15: number
  temp: number
  precip: number
  hour: number
  dayOfWeek: number  // 0=Monday, 6=Sunday
}

interface PredictionResult {
  prediction: number
  confidence: string
  surgeLevel: string
}

interface ModelInfoPayload {
  algorithm?: string
  n_estimators?: number
  features?: string[]
  metrics?: Record<string, number | null>
}

function formatApiError(payload: unknown): string {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const d = (payload as { detail: unknown }).detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) {
      const first = d[0]
      if (first && typeof first === 'object' && 'msg' in first) {
        return String((first as { msg: unknown }).msg)
      }
      return 'Invalid request — check inputs.'
    }
  }
  return 'Unable to complete prediction.'
}

function App() {
  const [input, setInput] = useState<PredictionInput>({
    supplyElasticity: 0.8,
    lagDER15: 1.2,
    lagDER30: 1.1,
    demandVelocity: 5,
    lagDemandVelocity15: 3,
    temp: 15,
    precip: 0,
    hour: 12,
    dayOfWeek: 4  // Friday (more surge!)
  })
  
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState<{
    loading: boolean
    data: ModelInfoPayload | null
  }>({ loading: true, data: null })

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/model-info')
        if (!res.ok) throw new Error('model-info failed')
        const data = (await res.json()) as ModelInfoPayload
        if (!cancelled) setModelInfo({ loading: false, data })
      } catch {
        if (!cancelled) setModelInfo({ loading: false, data: null })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const handleInputChange = (field: keyof PredictionInput, value: string) => {
    setInput(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }))
  }

  const getSurgeLevelColor = (level: string): string => {
    switch(level) {
      case 'Low Demand': return '#4ade80'
      case 'Normal': return '#60a5fa'
      case 'Moderate Surge': return '#fbbf24'
      case 'High Surge': return '#f97316'
      case 'Extreme Surge': return '#ef4444'
      default: return '#9ca3af'
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    
    try {
      // Same-origin when API serves the built SPA (Docker); use Vite proxy locally if needed.
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input)
      })
      
      const raw = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(formatApiError(raw) || `Error ${response.status}`)
      }
      const data = raw as {
        prediction: number
        confidence: string
        surgeLevel: string
      }
      setResult({
        prediction: data.prediction,
        confidence: data.confidence,
        surgeLevel: data.surgeLevel
      })
    } catch (error) {
      console.error('Prediction failed:', error)
      const base =
        error instanceof Error
          ? error.message
          : 'Could not reach the prediction service.'
      const hint = import.meta.env.DEV
        ? '\n\nLocal dev: run uvicorn on :8000 and `npm run dev` (Vite proxies /predict).'
        : ''
      alert(base + hint)
    } finally {
      setLoading(false)
    }
  }

  const handleLoadSample = (scenario: 'morning' | 'evening' | 'rainy') => {
    const samples = {
      morning: {
        supplyElasticity: 0.6,
        lagDER15: 1.8,
        lagDER30: 1.6,
        demandVelocity: 15,
        lagDemandVelocity15: 12,
        temp: 10,
        precip: 0,
        hour: 8,
        dayOfWeek: 1  // Tuesday morning commute
      },
      evening: {
        supplyElasticity: 0.4,
        lagDER15: 2.2,
        lagDER30: 2.0,
        demandVelocity: 25,
        lagDemandVelocity15: 20,
        temp: 18,
        precip: 0,
        hour: 18,
        dayOfWeek: 4  // Friday evening rush (highest surge!)
      },
      rainy: {
        supplyElasticity: 0.3,
        lagDER15: 2.5,
        lagDER30: 2.3,
        demandVelocity: 30,
        lagDemandVelocity15: 28,
        temp: 12,
        precip: 5.2,
        hour: 17,
        dayOfWeek: 5  // Saturday night + rain = extreme surge
      }
    }
    setInput(samples[scenario])
    setResult(null)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>NYC Taxi Surge Predictor</h1>
          <p>
            Interactive Demand Excess Ratio (DER) forecast from TLC-style market and weather
            features — powered by XGBoost
          </p>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <div className="grid">
            {/* Input Section */}
            <div className="card">
              <h2>Input Features</h2>
              
              <div className="sample-buttons">
                <button onClick={() => handleLoadSample('morning')} className="btn-sample">
                  🌅 Morning Rush
                </button>
                <button onClick={() => handleLoadSample('evening')} className="btn-sample">
                  🌆 Evening Rush
                </button>
                <button onClick={() => handleLoadSample('rainy')} className="btn-sample">
                  🌧️ Rainy Day
                </button>
              </div>

              <div className="form-section">
                <h3>Market Dynamics</h3>
                <div className="input-group">
                  <label>Supply Elasticity</label>
                  <input
                    type="number"
                    step="0.1"
                    value={input.supplyElasticity}
                    onChange={(e) => handleInputChange('supplyElasticity', e.target.value)}
                  />
                  <span className="hint">Ratio of drivers to requests</span>
                </div>

                <div className="input-group">
                  <label>DER (t-15 min)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={input.lagDER15}
                    onChange={(e) => handleInputChange('lagDER15', e.target.value)}
                  />
                  <span className="hint">Demand Excess Ratio 15 min ago</span>
                </div>

                <div className="input-group">
                  <label>DER (t-30 min)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={input.lagDER30}
                    onChange={(e) => handleInputChange('lagDER30', e.target.value)}
                  />
                  <span className="hint">Demand Excess Ratio 30 min ago</span>
                </div>

                <div className="input-group">
                  <label>Demand Velocity</label>
                  <input
                    type="number"
                    step="1"
                    value={input.demandVelocity}
                    onChange={(e) => handleInputChange('demandVelocity', e.target.value)}
                  />
                  <span className="hint">Rate of demand change</span>
                </div>

                <div className="input-group">
                  <label>Lag Demand Velocity</label>
                  <input
                    type="number"
                    step="1"
                    value={input.lagDemandVelocity15}
                    onChange={(e) => handleInputChange('lagDemandVelocity15', e.target.value)}
                  />
                  <span className="hint">Velocity 15 min ago</span>
                </div>
              </div>

              <div className="form-section">
                <h3>Weather & Time</h3>
                <div className="input-group">
                  <label>Temperature (°C)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={input.temp}
                    onChange={(e) => handleInputChange('temp', e.target.value)}
                  />
                </div>

                <div className="input-group">
                  <label>Precipitation (mm)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={input.precip}
                    onChange={(e) => handleInputChange('precip', e.target.value)}
                  />
                </div>

                <div className="input-group">
                  <label>Hour of Day</label>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={input.hour}
                    onChange={(e) => handleInputChange('hour', e.target.value)}
                  />
                  <span className="hint">0-23 (military time)</span>
                </div>

                <div className="input-group">
                  <label>Day of Week</label>
                  <select
                    value={input.dayOfWeek}
                    onChange={(e) => handleInputChange('dayOfWeek', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      border: '2px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '1rem',
                      backgroundColor: 'white'
                    }}
                  >
                    <option value="0">Monday</option>
                    <option value="1">Tuesday</option>
                    <option value="2">Wednesday</option>
                    <option value="3">Thursday</option>
                    <option value="4">Friday 🔥</option>
                    <option value="5">Saturday 🔥</option>
                    <option value="6">Sunday</option>
                  </select>
                  <span className="hint">Fridays/Saturdays surge more!</span>
                </div>
              </div>

              <button 
                className="btn-primary"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? 'Predicting...' : '🎯 Predict Surge'}
              </button>
            </div>

            {/* Results Section */}
            <div className="card">
              <h2>Prediction Results</h2>
              
              {result ? (
                <div className="results">
                  <div 
                    className="result-main"
                    style={{ 
                      borderColor: getSurgeLevelColor(result.surgeLevel),
                      backgroundColor: `${getSurgeLevelColor(result.surgeLevel)}15`
                    }}
                  >
                    <div className="result-label">Predicted Surge Level</div>
                    <div 
                      className="result-value"
                      style={{ color: getSurgeLevelColor(result.surgeLevel) }}
                    >
                      {result.surgeLevel}
                    </div>
                    <div className="result-der">
                      DER: {result.prediction.toFixed(2)}x
                    </div>
                  </div>

                  <div className="result-metrics">
                    <div className="metric">
                      <div className="metric-label">Model Confidence</div>
                      <div className="metric-value">{result.confidence}</div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Demand Excess Ratio</div>
                      <div className="metric-value">{result.prediction.toFixed(2)}x</div>
                    </div>
                  </div>

                  <div className="result-info">
                    <h3>📊 What this means:</h3>
                    {result.prediction < 0.8 && (
                      <p>Low demand period. More drivers than riders. Consider reducing active drivers.</p>
                    )}
                    {result.prediction >= 0.8 && result.prediction < 1.2 && (
                      <p>Balanced supply and demand. Normal operations expected.</p>
                    )}
                    {result.prediction >= 1.2 && result.prediction < 1.5 && (
                      <p>Moderate surge expected. Demand exceeding supply. Consider activating more drivers.</p>
                    )}
                    {result.prediction >= 1.5 && result.prediction < 2.0 && (
                      <p>High surge conditions. Significant demand pressure. Surge pricing recommended.</p>
                    )}
                    {result.prediction >= 2.0 && (
                      <p>Extreme surge! Very high demand with limited supply. Maximum surge pricing advised.</p>
                    )}
                  </div>

                  <div className="result-features">
                    <h3>Key Factors</h3>
                    <div className="feature-chips">
                      {input.precip > 1 && <span className="chip chip-weather">🌧️ Rain Impact</span>}
                      {input.hour >= 7 && input.hour <= 9 && <span className="chip chip-time">🌅 Morning Rush</span>}
                      {input.hour >= 17 && input.hour <= 19 && <span className="chip chip-time">🌆 Evening Rush</span>}
                      {input.demandVelocity > 20 && <span className="chip chip-demand">📈 High Velocity</span>}
                      {input.supplyElasticity < 0.5 && <span className="chip chip-supply">🚗 Low Supply</span>}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="results-empty">
                  <div className="empty-icon">📊</div>
                  <p>Enter features and click "Predict Surge" to see results</p>
                  <p className="empty-hint">Try loading a sample scenario to get started!</p>
                </div>
              )}
            </div>
          </div>

          {/* Info Section */}
          <div className="info-card">
            <h3>About this model</h3>
            <div className="info-grid">
              <div className="info-item">
                <strong>Algorithm</strong>
                {modelInfo.loading
                  ? 'Loading…'
                  : modelInfo.data?.algorithm ?? 'XGBoost Regressor'}
              </div>
              <div className="info-item">
                <strong>Training context</strong>
                Large-scale NYC TLC trip samples with Dask-based preprocessing (see repo README).
              </div>
              <div className="info-item">
                <strong>Test MAE</strong>
                {modelInfo.loading
                  ? 'Loading…'
                  : modelInfo.data?.metrics?.mae != null
                    ? `${modelInfo.data.metrics.mae.toFixed(4)} (holdout)`
                    : 'See model metadata after training'}
              </div>
              <div className="info-item">
                <strong>Feature count</strong>
                {modelInfo.loading
                  ? 'Loading…'
                  : modelInfo.data?.features?.length != null
                    ? `${modelInfo.data.features.length} (manifest from training)`
                    : 'Aligned with saved model_info.pkl'}
              </div>
            </div>
          </div>

          <footer className="demo-footer">
            <p>
              <strong>Demo notice:</strong> Outputs are for illustration and research. They are not
              financial or operational advice. Do not use this UI as the sole basis for pricing or
              dispatch decisions.
            </p>
            <p className="demo-footer-meta">
              API docs available at <code>/docs</code> when the backend is running.
            </p>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
