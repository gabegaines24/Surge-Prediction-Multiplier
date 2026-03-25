"""
Load project configuration from config.yaml with safe defaults.
"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULTS: Dict[str, Any] = {
    "paths": {
        "processed_dir": "./processed_data/",
        "model_dir": "./models/",
        "taxi_data_dir": "./taxi data/",
    },
    "model": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "reg:squarederror",
        "random_state": 42,
        "early_stopping_rounds": 20,
        "eval_metric": "mae",
    },
    "tuning": {
        "enabled": False,
        "cv_splits": 3,
        "max_samples": 20000,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        },
    },
    "data": {
        "train_test_split_date": "2025-02-15",
        "weather_start": "2025-01-01",
        "weather_end": "2025-03-31",
        "der_outlier_upper": 15.0,
        "der_outlier_lower": 0.01,
        "min_active_requests": 0,
    },
    "mlflow": {"enabled": False, "experiment_name": "surge_prediction"},
    "api": {"host": "0.0.0.0", "port": 8000, "predict_cache_maxsize": 256},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config from repo root; merge onto defaults."""
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    path = Path(path)
    cfg = deepcopy(_DEFAULTS)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user)
    # Allow env overrides for MLflow
    if os.environ.get("SURGE_MLFLOW_ENABLED", "").lower() in ("1", "true", "yes"):
        cfg["mlflow"]["enabled"] = True
    return cfg


def get_project_root() -> Path:
    return Path(__file__).resolve().parent
