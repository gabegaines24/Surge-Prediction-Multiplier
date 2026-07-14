import { useEffect, useState } from 'react'
import './App.css'
import { SliderField } from './SliderField'
import type {
  PredictionInput,
  PredictionResult,
  PredictionUncertainty,
  ShapExplanation,
} from './predictionTypes'
import { displayFeatureName } from './predictionTypes'
import {
  DEFAULT_UI_BOUNDS,
  clampPredictionInput,
  parseUiBounds,
  type UiBounds,
} from './uiConfig'

const POPULAR_TLC_ZONES = [161, 132, 138, 237, 234, 4, 43, 68, 186, 263]

interface ModelInfoPayload {
  algorithm?: string
  n_estimators?: number
  features?: string[]
  metrics?: Record<string, number | null>
}

function parseUncertainty(raw: unknown): PredictionUncertainty | null {
  if (!raw || typeof raw !== 'object') return null
  const u = raw as Record<string, unknown>
  const bandRaw = u.holdoutErrorBand
  let holdoutErrorBand: PredictionUncertainty['holdoutErrorBand']
  if (bandRaw && typeof bandRaw === 'object') {
    const b = bandRaw as Record<string, unknown>
    const lower = Number(b.lower)
    const upper = Number(b.upper)
    if (Number.isFinite(lower) && Number.isFinite(upper)) {
      holdoutErrorBand = {
        lower,
        upper,
        nominalLevel: typeof b.nominalLevel === 'number' ? b.nominalLevel : undefined,
        basis: typeof b.basis === 'string' ? b.basis : undefined,
        caveat: typeof b.caveat === 'string' ? b.caveat : undefined,
      }
    }
  }
  return {
    treeEnsembleDispersion:
      typeof u.treeEnsembleDispersion === 'string' ? u.treeEnsembleDispersion : undefined,
    note: typeof u.note === 'string' ? u.note : undefined,
    holdoutErrorBand,
  }
}

function parseShap(raw: unknown): ShapExplanation | null {
  if (!raw || typeof raw !== 'object') return null
  const s = raw as Record<string, unknown>
  const expectedValue = Number(s.expectedValue)
  const top = s.topFeatures
  if (!Number.isFinite(expectedValue) || !Array.isArray(top)) return null
  const topFeatures = top
    .map((item) => {
      if (!item || typeof item !== 'object') return null
      const f = item as Record<string, unknown>
      const feature = typeof f.feature === 'string' ? f.feature : null
      const shapValue = Number(f.shapValue)
      if (!feature || !Number.isFinite(shapValue)) return null
      return { feature, shapValue }
    })
    .filter((x): x is { feature: string; shapValue: number } => x != null)
  if (topFeatures.length === 0) return null
  return {
    expectedValue,
    topFeatures,
    method: typeof s.method === 'string' ? s.method : undefined,
  }
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

type ThemePref = 'light' | 'dark' | 'system'

const THEME_KEY = 'surge-pred-theme'

function App() {
  const [bounds, setBounds] = useState<UiBounds>(DEFAULT_UI_BOUNDS)
  const [input, setInput] = useState<PredictionInput>({
    zoneId: 161,
    supplyElasticity: 0.8,
    lagDER15: 1.2,
    lagDER30: 1.1,
    demandVelocity: 5,
    lagDemandVelocity15: 3,
    temp: 15,
    precip: 0,
    hour: 12,
    dayOfWeek: 4,
  })

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState<{
    loading: boolean
    data: ModelInfoPayload | null
  }>({ loading: true, data: null })

  const [themePref, setThemePref] = useState<ThemePref>(() => {
    try {
      const s = localStorage.getItem(THEME_KEY) as ThemePref | null
      if (s === 'light' || s === 'dark' || s === 'system') return s
    } catch {
      /* ignore */
    }
    return 'system'
  })

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/config/ui')
        if (!res.ok) throw new Error('config/ui failed')
        const raw = await res.json()
        if (!cancelled) setBounds(parseUiBounds(raw))
      } catch {
        if (!cancelled) setBounds(DEFAULT_UI_BOUNDS)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    setInput((prev) => clampPredictionInput(prev, bounds))
  }, [bounds])

  useEffect(() => {
    try {
      localStorage.setItem(THEME_KEY, themePref)
    } catch {
      /* ignore */
    }
  }, [themePref])

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => {
      let resolved: 'light' | 'dark' = 'light'
      if (themePref === 'dark') resolved = 'dark'
      else if (themePref === 'light') resolved = 'light'
      else resolved = mq.matches ? 'dark' : 'light'
      document.documentElement.dataset.theme = resolved
    }
    apply()
    if (themePref === 'system') {
      mq.addEventListener('change', apply)
      return () => mq.removeEventListener('change', apply)
    }
  }, [themePref])

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

  const setField = (partial: Partial<PredictionInput>) => {
    setInput((prev) => clampPredictionInput({ ...prev, ...partial }, bounds))
  }

  const getSurgeLevelColor = (level: string): string => {
    switch (level) {
      case 'Low Demand':
        return 'var(--surge-low)'
      case 'Normal':
        return 'var(--surge-normal)'
      case 'Moderate Surge':
        return 'var(--surge-moderate)'
      case 'High Surge':
        return 'var(--surge-high)'
      case 'Extreme Surge':
        return 'var(--surge-extreme)'
      default:
        return 'var(--text-muted)'
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    setResult(null)
    const payload = clampPredictionInput(input, bounds)
    const body = JSON.stringify(payload)
    const headers = { 'Content-Type': 'application/json' }
    try {
      const [predictRes, shapRes] = await Promise.all([
        fetch('/predict', { method: 'POST', headers, body }),
        fetch('/interpretability/shap?max_features=8', { method: 'POST', headers, body }),
      ])

      const predictRaw = await predictRes.json().catch(() => ({}))
      if (!predictRes.ok) {
        throw new Error(formatApiError(predictRaw) || `Error ${predictRes.status}`)
      }

      const prediction = Number((predictRaw as { prediction?: unknown }).prediction)
      const confidence = String((predictRaw as { confidence?: unknown }).confidence ?? '')
      const surgeLevel = String((predictRaw as { surgeLevel?: unknown }).surgeLevel ?? '')
      if (!Number.isFinite(prediction) || !surgeLevel) {
        throw new Error('Unexpected prediction response.')
      }

      let shap: ShapExplanation | null = null
      let shapError: string | null = null
      if (shapRes.ok) {
        const shapRaw = await shapRes.json().catch(() => null)
        shap = parseShap(shapRaw)
        if (!shap) shapError = 'SHAP explanation could not be parsed.'
      } else {
        const shapRaw = await shapRes.json().catch(() => ({}))
        shapError = formatApiError(shapRaw) || `SHAP unavailable (${shapRes.status})`
      }

      setResult({
        prediction,
        confidence,
        surgeLevel,
        uncertainty: parseUncertainty((predictRaw as { uncertainty?: unknown }).uncertainty),
        shap,
        shapError,
      })
    } catch (error) {
      console.error('Prediction failed:', error)
      const base =
        error instanceof Error ? error.message : 'Could not reach the prediction service.'
      const hint = import.meta.env.DEV
        ? '\n\nLocal dev: run uvicorn on :8000 and `npm run dev` (Vite proxies /predict).'
        : ''
      alert(base + hint)
    } finally {
      setLoading(false)
    }
  }

  const handleLoadSample = (scenario: 'morning' | 'evening' | 'rainy') => {
    const samples: Record<string, PredictionInput> = {
      morning: {
        zoneId: 186,
        supplyElasticity: 0.6,
        lagDER15: 1.8,
        lagDER30: 1.6,
        demandVelocity: 15,
        lagDemandVelocity15: 12,
        temp: 10,
        precip: 0,
        hour: 8,
        dayOfWeek: 1,
      },
      evening: {
        zoneId: 161,
        supplyElasticity: 0.4,
        lagDER15: 2.2,
        lagDER30: 2.0,
        demandVelocity: 25,
        lagDemandVelocity15: 20,
        temp: 18,
        precip: 0,
        hour: 18,
        dayOfWeek: 4,
      },
      rainy: {
        zoneId: 132,
        supplyElasticity: 0.3,
        lagDER15: 2.5,
        lagDER30: 2.3,
        demandVelocity: 30,
        lagDemandVelocity15: 28,
        temp: 12,
        precip: 5.2,
        hour: 17,
        dayOfWeek: 5,
      },
    }
    setInput(clampPredictionInput(samples[scenario], bounds))
    setResult(null)
  }

  const hb = bounds.hour
  const hourBounds = { min: hb.min, max: hb.max, step: 1 }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content header-row">
          <div>
            <h1>NYC Taxi Surge Predictor</h1>
            <p>
              Interactive Demand Excess Ratio (DER) forecast from TLC-style market and weather
              features — powered by XGBoost
            </p>
          </div>
          <div className="theme-toggle" role="group" aria-label="Theme">
            <button
              type="button"
              className={themePref === 'light' ? 'theme-btn active' : 'theme-btn'}
              onClick={() => setThemePref('light')}
            >
              Light
            </button>
            <button
              type="button"
              className={themePref === 'dark' ? 'theme-btn active' : 'theme-btn'}
              onClick={() => setThemePref('dark')}
            >
              Dark
            </button>
            <button
              type="button"
              className={themePref === 'system' ? 'theme-btn active' : 'theme-btn'}
              onClick={() => setThemePref('system')}
            >
              System
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <div className="grid">
            <div className="card">
              <h2>Input Features</h2>

              <div className="sample-buttons">
                <button type="button" onClick={() => handleLoadSample('morning')} className="btn-sample">
                  Morning rush
                </button>
                <button type="button" onClick={() => handleLoadSample('evening')} className="btn-sample">
                  Evening rush
                </button>
                <button type="button" onClick={() => handleLoadSample('rainy')} className="btn-sample">
                  Rainy day
                </button>
              </div>

              <div className="form-section">
                <h3>Location &amp; market</h3>
                <div className="input-group">
                  <label htmlFor="zoneId">TLC zone ID</label>
                  <input
                    id="zoneId"
                    type="number"
                    min={bounds.zoneId.min}
                    max={bounds.zoneId.max}
                    step={1}
                    list="popular-tlc-zones"
                    value={input.zoneId}
                    onChange={(e) => {
                      const v = parseInt(e.target.value, 10)
                      setField({ zoneId: Number.isFinite(v) ? v : bounds.zoneId.min })
                    }}
                  />
                  <datalist id="popular-tlc-zones">
                    {POPULAR_TLC_ZONES.map((z) => (
                      <option key={z} value={z} />
                    ))}
                  </datalist>
                  <span className="hint">NYC TLC location IDs {bounds.zoneId.min}–{bounds.zoneId.max}</span>
                </div>

                <SliderField
                  id="supplyElasticity"
                  label="Supply elasticity"
                  hint="Ratio of drivers to requests"
                  bounds={bounds.supplyElasticity}
                  value={input.supplyElasticity}
                  onChange={(v) => setField({ supplyElasticity: v })}
                />
                <SliderField
                  id="lagDER15"
                  label="DER (t−15 min)"
                  hint="Demand excess ratio 15 min ago"
                  bounds={bounds.lagDER15}
                  value={input.lagDER15}
                  onChange={(v) => setField({ lagDER15: v })}
                />
                <SliderField
                  id="lagDER30"
                  label="DER (t−30 min)"
                  hint="Demand excess ratio 30 min ago"
                  bounds={bounds.lagDER30}
                  value={input.lagDER30}
                  onChange={(v) => setField({ lagDER30: v })}
                />
                <SliderField
                  id="demandVelocity"
                  label="Demand velocity"
                  hint="Rate of demand change"
                  bounds={bounds.demandVelocity}
                  value={input.demandVelocity}
                  onChange={(v) => setField({ demandVelocity: v })}
                />
                <SliderField
                  id="lagDemandVelocity15"
                  label="Lag demand velocity"
                  hint="Velocity 15 min ago"
                  bounds={bounds.lagDemandVelocity15}
                  value={input.lagDemandVelocity15}
                  onChange={(v) => setField({ lagDemandVelocity15: v })}
                />
              </div>

              <div className="form-section">
                <h3>Weather &amp; time</h3>
                <SliderField
                  id="temp"
                  label="Temperature (°C)"
                  bounds={bounds.temp}
                  value={input.temp}
                  onChange={(v) => setField({ temp: v })}
                />
                <SliderField
                  id="precip"
                  label="Precipitation (mm)"
                  bounds={bounds.precip}
                  value={input.precip}
                  onChange={(v) => setField({ precip: v })}
                />
                <SliderField
                  id="hour"
                  label="Hour of day"
                  hint="0–23"
                  bounds={hourBounds}
                  value={input.hour}
                  onChange={(v) => setField({ hour: Math.round(v) })}
                />
                <div className="input-group">
                  <label htmlFor="dayOfWeek">Day of week</label>
                  <select
                    id="dayOfWeek"
                    value={input.dayOfWeek}
                    onChange={(e) => setField({ dayOfWeek: parseInt(e.target.value, 10) })}
                    className="select-input"
                  >
                    <option value="0">Monday</option>
                    <option value="1">Tuesday</option>
                    <option value="2">Wednesday</option>
                    <option value="3">Thursday</option>
                    <option value="4">Friday</option>
                    <option value="5">Saturday</option>
                    <option value="6">Sunday</option>
                  </select>
                  <span className="hint">Weekend and evening patterns differ</span>
                </div>
              </div>

              <button type="button" className="btn-primary" onClick={handlePredict} disabled={loading}>
                {loading ? 'Predicting…' : 'Predict surge'}
              </button>
            </div>

            <div className="card">
              <h2>Prediction Results</h2>

              {result ? (
                <div className="results">
                  <div
                    className="result-main"
                    style={{
                      borderColor: getSurgeLevelColor(result.surgeLevel),
                    }}
                  >
                    <div className="result-label">Predicted surge level</div>
                    <div className="result-value" style={{ color: getSurgeLevelColor(result.surgeLevel) }}>
                      {result.surgeLevel}
                    </div>
                    <div className="result-der">DER: {result.prediction.toFixed(2)}x</div>
                    {result.uncertainty?.holdoutErrorBand && (
                      <div className="result-band">
                        Rough band:{' '}
                        {result.uncertainty.holdoutErrorBand.lower.toFixed(2)}x –{' '}
                        {result.uncertainty.holdoutErrorBand.upper.toFixed(2)}x
                        {result.uncertainty.holdoutErrorBand.nominalLevel != null && (
                          <span className="result-band-level">
                            {' '}
                            (~{Math.round(result.uncertainty.holdoutErrorBand.nominalLevel * 100)}%
                            holdout residuals)
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="result-metrics">
                    <div className="metric">
                      <div className="metric-label">Model confidence</div>
                      <div className="metric-value">{result.confidence}</div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Demand excess ratio</div>
                      <div className="metric-value">{result.prediction.toFixed(2)}x</div>
                    </div>
                  </div>

                  {(result.uncertainty?.holdoutErrorBand ||
                    result.uncertainty?.treeEnsembleDispersion) && (
                    <div className="result-uncertainty">
                      <h3>Uncertainty</h3>
                      {result.uncertainty.holdoutErrorBand && (
                        <p>
                          Empirical residual band around the point forecast:{' '}
                          <strong>
                            {result.uncertainty.holdoutErrorBand.lower.toFixed(2)}x –{' '}
                            {result.uncertainty.holdoutErrorBand.upper.toFixed(2)}x
                          </strong>
                          . Heuristic from holdout residuals, not a calibrated prediction interval.
                        </p>
                      )}
                      {result.uncertainty.treeEnsembleDispersion && (
                        <p className="uncertainty-dispersion">
                          Tree-ensemble dispersion proxy:{' '}
                          <strong>{result.uncertainty.treeEnsembleDispersion}</strong>
                          {result.uncertainty.note ? ` — ${result.uncertainty.note}` : ''}
                        </p>
                      )}
                    </div>
                  )}

                  <div className="result-info">
                    <h3>What this means</h3>
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
                      <p>Extreme surge: very high demand with limited supply. Maximum surge pricing advised.</p>
                    )}
                  </div>

                  <div className="result-shap">
                    <h3>Why this prediction?</h3>
                    {result.shap ? (
                      <>
                        <p className="shap-intro">
                          Top {result.shap.method ?? 'TreeSHAP'} drivers vs model baseline (
                          {result.shap.expectedValue.toFixed(2)}x). Bars push DER up or down from that
                          baseline.
                        </p>
                        <ul className="shap-list">
                          {(() => {
                            const maxAbs = Math.max(
                              ...result.shap.topFeatures.map((f) => Math.abs(f.shapValue)),
                              1e-9,
                            )
                            return result.shap.topFeatures.map((f) => {
                              const pct = (Math.abs(f.shapValue) / maxAbs) * 100
                              const up = f.shapValue >= 0
                              return (
                                <li key={f.feature} className="shap-row">
                                  <div className="shap-row-label">
                                    <span>{displayFeatureName(f.feature)}</span>
                                    <span className={up ? 'shap-val-up' : 'shap-val-down'}>
                                      {up ? '+' : ''}
                                      {f.shapValue.toFixed(3)}
                                    </span>
                                  </div>
                                  <div className="shap-bar-track" aria-hidden>
                                    <div
                                      className={up ? 'shap-bar shap-bar-up' : 'shap-bar shap-bar-down'}
                                      style={{ width: `${pct}%` }}
                                    />
                                  </div>
                                </li>
                              )
                            })
                          })()}
                        </ul>
                      </>
                    ) : (
                      <p className="shap-fallback">
                        {result.shapError ?? 'Local explanation unavailable for this run.'}
                      </p>
                    )}
                  </div>

                  <div className="result-features">
                    <h3>Key factors</h3>
                    <div className="feature-chips">
                      {input.precip > 1 && <span className="chip chip-weather">Rain impact</span>}
                      {input.hour >= 7 && input.hour <= 9 && <span className="chip chip-time">Morning rush</span>}
                      {input.hour >= 17 && input.hour <= 19 && <span className="chip chip-time">Evening rush</span>}
                      {input.demandVelocity > 20 && <span className="chip chip-demand">High velocity</span>}
                      {input.supplyElasticity < 0.5 && <span className="chip chip-supply">Low supply</span>}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="results-empty">
                  <div className="empty-icon" aria-hidden>
                    ◎
                  </div>
                  <p>Enter features and click Predict surge</p>
                  <p className="empty-hint">Try a sample scenario to get started</p>
                </div>
              )}
            </div>
          </div>

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
              <strong>Demo notice:</strong> Outputs are for illustration and research. They are not financial or
              operational advice. Do not use this UI as the sole basis for pricing or dispatch decisions.
            </p>
            <p className="demo-footer-meta">
              API docs at <code>/docs</code>; UI bounds from <code>/config/ui</code>. Predictions include
              residual uncertainty bands; local TreeSHAP explanations come from{' '}
              <code>/interpretability/shap</code>.
            </p>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
