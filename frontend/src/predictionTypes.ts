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
