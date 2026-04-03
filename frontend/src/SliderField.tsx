import type { NumericBounds } from './uiConfig'

type Props = {
  id: string
  label: string
  hint?: string
  bounds: NumericBounds
  value: number
  onChange: (v: number) => void
}

export function SliderField({ id, label, hint, bounds, value, onChange }: Props) {
  const { min, max, step } = bounds
  return (
    <div className="input-group">
      <label htmlFor={id}>{label}</label>
      <div className="slider-row">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <output className="slider-value" htmlFor={id}>
          {step >= 1
            ? String(Math.round(value))
            : step < 0.1
              ? value.toFixed(2)
              : value.toFixed(1)}
        </output>
      </div>
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  )
}
