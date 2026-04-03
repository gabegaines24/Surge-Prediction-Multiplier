"""Global feature importance and optional SHAP explanations for XGBoost."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

_shap_explainer: Any = None
_shap_model_id: str | None = None


def global_gain_importance(model: xgb.XGBRegressor) -> list[dict[str, Any]]:
    """
    XGBoost 'gain' importance, normalized to sum to 1.0 for comparability.
    Keys match training column names when available.
    """
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    if not raw:
        return []

    names = getattr(model, "feature_names_in_", None)
    mapped: dict[str, float] = {}
    for key, val in raw.items():
        if key.startswith("f") and names is not None:
            try:
                idx = int(key[1:])
                fname = str(names[idx]) if idx < len(names) else key
            except ValueError:
                fname = key
        else:
            fname = key
        mapped[fname] = float(val)

    total = sum(mapped.values()) or 1.0
    ranked = sorted(mapped.items(), key=lambda x: -x[1])
    return [{"feature": k, "gain": v, "gainFraction": v / total} for k, v in ranked]


def shap_explanation_row(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    *,
    model_mtime: float,
    max_features: int = 12,
) -> dict[str, Any]:
    """
    TreeSHAP values for a single row (n_samples=1).
    Returns top contributing features by absolute SHAP value.
    """
    try:
        import shap
    except ImportError as e:
        raise RuntimeError(
            "Install the 'shap' package for SHAP explanations: pip install shap>=0.45"
        ) from e

    global _shap_explainer, _shap_model_id
    mid = f"{id(model)}:{model_mtime:.6f}"
    if _shap_explainer is None or _shap_model_id != mid:
        _shap_explainer = shap.TreeExplainer(model)
        _shap_model_id = mid

    explainer = _shap_explainer
    sv = explainer.shap_values(X)
    base = explainer.expected_value
    base_f = float(np.asarray(base).reshape(-1)[0])
    if isinstance(sv, list):
        sv = sv[0]
    m = np.asarray(sv)
    if m.ndim == 2:
        m = m[0]
    sv_row = m.reshape(-1)
    names = list(getattr(model, "feature_names_in_", []))
    if len(names) != len(sv_row):
        names = [f"f{i}" for i in range(len(sv_row))]

    pairs = sorted(
        zip(names, sv_row.tolist()),
        key=lambda x: -abs(x[1]),
    )[:max_features]

    return {
        "expectedValue": base_f,
        "topFeatures": [{"feature": a, "shapValue": float(b)} for a, b in pairs],
        "method": "TreeSHAP",
    }
