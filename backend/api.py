"""
FastAPI service for surge prediction (replaces legacy Flask app).
Loads feature order from model_info.pkl so training/API stay aligned.
"""

from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.staticfiles import StaticFiles

from .config_loader import load_config
from .feature_engineering import add_calendar_from_hour_dow
from .zone_metadata import add_zone_hint_features

_cfg = load_config()
MODEL_DIR = _cfg["paths"]["model_dir"]
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_surge_model.pkl")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.pkl")

model = None
feature_names: list[str] = []
model_mtime: float = 0.0

# Response-level cache (short TTL); bounded by config
_predict_cache: TTLCache = TTLCache(
    maxsize=int(_cfg["api"].get("predict_cache_maxsize", 256)),
    ttl=120,
)


def load_model() -> None:
    """Load model + feature manifest from disk."""
    global model, feature_names, model_mtime
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   Run `python -m backend.train_model` first.")
        model = None
        feature_names = []
        model_mtime = 0.0
        return
    model = joblib.load(MODEL_PATH)
    model_mtime = os.path.getmtime(MODEL_PATH)
    if os.path.isfile(INFO_PATH):
        info = joblib.load(INFO_PATH)
        feature_names = list(info.get("features", []))
    else:
        print("⚠️  model_info.pkl missing — falling back to model.feature_names_in_")
        feature_names = list(getattr(model, "feature_names_in_", []))
    print(f"✓ Model loaded from {MODEL_PATH} ({len(feature_names)} features)")


app = FastAPI(title="NYC Taxi Surge Prediction API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    load_model()


class PredictRequest(BaseModel):
    """Request body (camelCase for frontend compatibility)."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    supply_elasticity: float = Field(..., alias="supplyElasticity")
    lag_der_15: float = Field(..., alias="lagDER15")
    lag_der_30: float = Field(..., alias="lagDER30")
    demand_velocity: float = Field(..., alias="demandVelocity")
    lag_demand_velocity_15: float = Field(..., alias="lagDemandVelocity15")
    temp: float = Field(..., alias="temp")
    precip: float = Field(..., alias="precip")
    hour: int = Field(..., ge=0, le=23, alias="hour")
    day_of_week: int = Field(..., ge=0, le=6, alias="dayOfWeek")

    zone_id: int = Field(161, alias="zoneId")
    month: Optional[int] = Field(None, ge=1, le=12, alias="month")
    is_holiday: Optional[bool] = Field(None, alias="isHoliday")
    der_rolling_mean_1h: Optional[float] = Field(None, alias="derRollingMean1h")
    der_rolling_std_1h: Optional[float] = Field(None, alias="derRollingStd1h")

    @field_validator("hour", "day_of_week", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        return int(v)

    @field_validator(
        "supply_elasticity",
        "lag_der_15",
        "lag_der_30",
        "demand_velocity",
        "lag_demand_velocity_15",
        "temp",
        "precip",
        "der_rolling_mean_1h",
        "der_rolling_std_1h",
        mode="before",
    )
    @classmethod
    def _validate_finite_float(cls, v: Any) -> Any:
        if v is None:
            return None
        val = float(v)
        if not math.isfinite(val):
            raise ValueError("must be a finite number")
        return val


def _inference_feature_dict(req: PredictRequest) -> tuple[dict[str, float], list[str]]:
    """Single-row feature dictionary aligned with training column names."""
    defaults_used: list[str] = []
    if req.month is None:
        defaults_used.append("month")
    if req.is_holiday is None:
        defaults_used.append("isHoliday")
    if req.der_rolling_mean_1h is None:
        defaults_used.append("derRollingMean1h")
    if req.der_rolling_std_1h is None:
        defaults_used.append("derRollingStd1h")

    cal = add_calendar_from_hour_dow(
        req.hour,
        req.day_of_week,
        month=req.month,
        is_holiday=(1 if req.is_holiday else 0) if req.is_holiday is not None else None,
    )
    z = add_zone_hint_features(pd.DataFrame({"Zone": [req.zone_id]}))
    roll_m = req.der_rolling_mean_1h if req.der_rolling_mean_1h is not None else req.lag_der_15
    roll_s = req.der_rolling_std_1h if req.der_rolling_std_1h is not None else 0.0

    d: dict[str, float] = {
        "SupplyElasticity": float(req.supply_elasticity),
        "Lag_DER_t-15": float(req.lag_der_15),
        "Lag_DER_t-30": float(req.lag_der_30),
        "DemandVelocity_t": float(req.demand_velocity),
        "Lag_DemandVelocity_t-15": float(req.lag_demand_velocity_15),
        "temp": float(req.temp),
        "precip": float(req.precip),
        "hour_sin": float(np.sin(2 * np.pi * req.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * req.hour / 24)),
        "day_of_week": float(req.day_of_week),
        "is_weekend": float(1 if req.day_of_week >= 5 else 0),
        "DER_rolling_mean_1h": float(roll_m),
        "DER_rolling_std_1h": float(roll_s),
        "month": float(cal["month"]),
        "month_sin": cal["month_sin"],
        "month_cos": cal["month_cos"],
        "is_rush_hour": float(cal["is_rush_hour"]),
        "is_holiday": float(cal["is_holiday"]),
        "is_airport_zone": float(z["is_airport_zone"].iloc[0]),
        "is_manhattan_core": float(z["is_manhattan_core"].iloc[0]),
        "Zone": float(req.zone_id),
    }
    return d, defaults_used


def _build_matrix(req: PredictRequest) -> tuple[np.ndarray, list[str]]:
    if not feature_names:
        raise HTTPException(
            status_code=500,
            detail="Model metadata missing feature list. Retrain with `python -m backend.train_model`.",
        )
    values, defaults_used = _inference_feature_dict(req)
    row: list[float] = []
    missing: list[str] = []
    for name in feature_names:
        if name in values:
            row.append(float(values[name]))
        else:
            missing.append(name)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot satisfy features (add to request or retrain): {missing[:8]}",
        )
    return np.array([row], dtype=np.float32), defaults_used


def _cache_key_for_matrix(X: np.ndarray) -> str:
    h = hashlib.sha256(X.tobytes()).hexdigest()
    return f"{model_mtime:.6f}:{h}"


def _tree_contribution_confidence(m: xgb.XGBRegressor, X: np.ndarray) -> float:
    """Use std dev of per-tree marginal predictions as an uncertainty proxy."""
    try:
        booster = m.get_booster()
        dm = xgb.DMatrix(X)
        n = booster.num_boosted_rounds()
        if n <= 1:
            return 0.85
        per_tree = [booster.predict(dm, iteration_range=(i, i + 1)) for i in range(n)]
        arr = np.vstack(per_tree).T
        std = float(np.std(arr[0]))
        conf = 1.0 / (1.0 + std)
        return max(0.15, min(0.98, conf))
    except Exception:
        return 0.80


def get_surge_level(der: float) -> str:
    if der < 0.8:
        return "Low Demand"
    if der < 1.2:
        return "Normal"
    if der < 1.5:
        return "Moderate Surge"
    if der < 2.0:
        return "High Surge"
    return "Extreme Surge"


def get_recommendation(der: float, precip: float) -> str:
    if der < 0.8:
        return "Supply exceeds demand. Consider reducing active driver incentives."
    if der < 1.2:
        return "Balanced market conditions. Maintain current operations."
    if der < 1.5:
        return "Moderate demand pressure. Consider 1.2-1.4x surge pricing."
    if der < 2.0:
        action = "Activate surge pricing (1.5-1.8x) and send driver alerts."
        if precip > 1:
            action += " Rain detected - expect sustained high demand."
        return action
    return "Extreme surge conditions. Implement maximum pricing (2.0x+) and emergency driver activation."


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Run `python -m backend.train_model` first.",
        )
    X, defaults_used = _build_matrix(body)
    ck = _cache_key_for_matrix(X)
    if ck in _predict_cache:
        return dict(_predict_cache[ck])

    prediction = float(model.predict(X)[0])
    confidence = _tree_contribution_confidence(model, X)
    surge = get_surge_level(prediction)
    out = {
        "prediction": prediction,
        "surgeLevel": surge,
        "confidence": f"{confidence:.0%}",
        "recommendation": get_recommendation(prediction, body.precip),
        "defaultsUsed": defaults_used,
        "timestamp": datetime.now().isoformat(),
    }
    _predict_cache[ck] = out
    return out


def _json_safe_metrics(m: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    info_extra: dict[str, Any] = {}
    if os.path.isfile(INFO_PATH):
        info_extra = joblib.load(INFO_PATH)
    return {
        "algorithm": "XGBoost",
        "n_estimators": int(model.n_estimators),
        "max_depth": int(model.max_depth),
        "learning_rate": float(model.learning_rate),
        "features": feature_names or list(getattr(model, "feature_names_in_", [])),
        "metrics": _json_safe_metrics(dict(info_extra.get("metrics", {}))),
        "artifact": info_extra.get("artifact_version", MODEL_PATH),
    }


# React production build (Docker / Hugging Face). Mounted last so API routes win.
_FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend_dist")
)
_index_html = os.path.join(_FRONTEND_DIST, "index.html")
if os.path.isfile(_index_html):
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND_DIST, html=True),
        name="spa",
    )

# Legacy Flask-style entry removed; run with:
#   uvicorn backend.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    host = str(_cfg["api"].get("host", "0.0.0.0"))
    port = int(_cfg["api"].get("port", 8000))
    print("=" * 60)
    print("NYC Taxi Surge Prediction API (FastAPI)")
    print("=" * 60)
    load_model()
    print("\nEndpoints:")
    print("  GET  /health      - Health check")
    print("  POST /predict     - Make prediction")
    print("  GET  /model-info  - Model information")
    print(f"\nStarting server on http://{host}:{port}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
