/** Request body for POST /predict (camelCase, matches FastAPI aliases). */

export interface PredictionInput {
  zoneId: number
  supplyElasticity: number
  lagDER15: number
  lagDER30: number
  demandVelocity: number
  lagDemandVelocity15: number
  temp: number
  precip: number
  hour: number
  dayOfWeek: number
}

/** Holdout residual band from POST /predict `uncertainty.holdoutErrorBand`. */
export interface HoldoutErrorBand {
  lower: number
  upper: number
  nominalLevel?: number
  basis?: string
  caveat?: string
}

export interface PredictionUncertainty {
  treeEnsembleDispersion?: string
  note?: string
  holdoutErrorBand?: HoldoutErrorBand
}

export interface ShapFeatureContribution {
  feature: string
  shapValue: number
}

/** Response from POST /interpretability/shap. */
export interface ShapExplanation {
  expectedValue: number
  topFeatures: ShapFeatureContribution[]
  method?: string
}

export interface PredictionResult {
  prediction: number
  confidence: string
  surgeLevel: string
  uncertainty?: PredictionUncertainty | null
  shap?: ShapExplanation | null
  shapError?: string | null
}

/** Friendlier labels for TreeSHAP / training column names. */
export const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  DER_rolling_mean_1h: 'DER rolling mean (1h)',
  DER_rolling_std_1h: 'DER rolling std (1h)',
  DemandVelocity_t: 'Demand velocity',
  'Lag_DER_t-15': 'DER lag (t−15)',
  'Lag_DER_t-30': 'DER lag (t−30)',
  'Lag_DemandVelocity_t-15': 'Velocity lag (t−15)',
  SupplyElasticity: 'Supply elasticity',
  Zone: 'TLC zone',
  day_of_week: 'Day of week',
  hour_cos: 'Hour (cos)',
  hour_sin: 'Hour (sin)',
  is_airport_zone: 'Airport zone',
  is_holiday: 'Holiday',
  is_manhattan_core: 'Manhattan core',
  is_rush_hour: 'Rush hour',
  is_weekend: 'Weekend',
  month: 'Month',
  month_cos: 'Month (cos)',
  month_sin: 'Month (sin)',
  precip: 'Precipitation',
  temp: 'Temperature',
}

export function displayFeatureName(name: string): string {
  return FEATURE_DISPLAY_NAMES[name] ?? name.replace(/_/g, ' ')
}
