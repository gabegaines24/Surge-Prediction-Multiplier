/** Bounds aligned with GET /config/ui and backend PredictRequest. */

import type { PredictionInput } from './predictionTypes'

export type NumericBounds = { min: number; max: number; step: number }
export type IntBounds = { min: number; max: number }

export type UiBounds = {
  zoneId: IntBounds
  hour: IntBounds
  dayOfWeek: IntBounds
  supplyElasticity: NumericBounds
  lagDER15: NumericBounds
  lagDER30: NumericBounds
  demandVelocity: NumericBounds
  lagDemandVelocity15: NumericBounds
  temp: NumericBounds
  precip: NumericBounds
}

export const DEFAULT_UI_BOUNDS: UiBounds = {
  zoneId: { min: 1, max: 263 },
  hour: { min: 0, max: 23 },
  dayOfWeek: { min: 0, max: 6 },
  supplyElasticity: { min: 0, max: 10, step: 0.05 },
  lagDER15: { min: 0.01, max: 15, step: 0.05 },
  lagDER30: { min: 0.01, max: 15, step: 0.05 },
  demandVelocity: { min: -200, max: 200, step: 1 },
  lagDemandVelocity15: { min: -200, max: 200, step: 1 },
  temp: { min: -30, max: 45, step: 0.5 },
  precip: { min: 0, max: 50, step: 0.1 },
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}

export function parseUiBounds(raw: unknown): UiBounds {
  if (!raw || typeof raw !== 'object') return DEFAULT_UI_BOUNDS
  const o = raw as Record<string, { min?: number; max?: number; step?: number }>
  const nb = (
    k:
      | 'supplyElasticity'
      | 'lagDER15'
      | 'lagDER30'
      | 'demandVelocity'
      | 'lagDemandVelocity15'
      | 'temp'
      | 'precip'
  ): NumericBounds => {
    const x = o[k]
    const d = DEFAULT_UI_BOUNDS[k]
    if (!x || typeof x.min !== 'number' || typeof x.max !== 'number') return d
    return {
      min: x.min,
      max: x.max,
      step: typeof x.step === 'number' ? x.step : d.step,
    }
  }
  const ib = (k: 'zoneId' | 'hour' | 'dayOfWeek'): IntBounds => {
    const x = o[k]
    const d = DEFAULT_UI_BOUNDS[k]
    if (!x || typeof x.min !== 'number' || typeof x.max !== 'number') return d
    return { min: x.min, max: x.max }
  }
  return {
    zoneId: ib('zoneId'),
    hour: ib('hour'),
    dayOfWeek: ib('dayOfWeek'),
    supplyElasticity: nb('supplyElasticity'),
    lagDER15: nb('lagDER15'),
    lagDER30: nb('lagDER30'),
    demandVelocity: nb('demandVelocity'),
    lagDemandVelocity15: nb('lagDemandVelocity15'),
    temp: nb('temp'),
    precip: nb('precip'),
  }
}

export function clampPredictionInput(
  input: PredictionInput,
  b: UiBounds
): PredictionInput {
  return {
    zoneId: Math.round(clamp(input.zoneId, b.zoneId.min, b.zoneId.max)),
    supplyElasticity: clamp(input.supplyElasticity, b.supplyElasticity.min, b.supplyElasticity.max),
    lagDER15: clamp(input.lagDER15, b.lagDER15.min, b.lagDER15.max),
    lagDER30: clamp(input.lagDER30, b.lagDER30.min, b.lagDER30.max),
    demandVelocity: clamp(input.demandVelocity, b.demandVelocity.min, b.demandVelocity.max),
    lagDemandVelocity15: clamp(
      input.lagDemandVelocity15,
      b.lagDemandVelocity15.min,
      b.lagDemandVelocity15.max
    ),
    temp: clamp(input.temp, b.temp.min, b.temp.max),
    precip: clamp(input.precip, b.precip.min, b.precip.max),
    hour: Math.round(clamp(input.hour, b.hour.min, b.hour.max)),
    dayOfWeek: Math.round(clamp(input.dayOfWeek, b.dayOfWeek.min, b.dayOfWeek.max)),
  }
}
