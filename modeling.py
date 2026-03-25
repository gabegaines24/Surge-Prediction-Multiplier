from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def prepare_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert hour to sine/cosine so 23:00 is close to 00:00."""
    df = df.copy()
    df["hour"] = df["Time_Bin"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def _mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "mape": _mean_absolute_percentage_error(yt, yp),
    }


def time_series_cv_scores(
    X: pd.DataFrame,
    y: pd.Series,
    base_params: Mapping[str, Any],
    n_splits: int = 5,
) -> Dict[str, float]:
    """Mean MAE across time-ordered CV folds (no early stopping per fold)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_maes: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBRegressor(**base_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        fold_maes.append(mean_absolute_error(y.iloc[val_idx], preds))
    return {
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
    }


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Optional grid search with TimeSeriesSplit (can be slow on large data).
    Uses a row subsample for speed when tuning.max_samples is set.
    """
    tcfg = config["tuning"]
    max_samples = int(tcfg.get("max_samples", 20_000))
    if len(X) > max_samples:
        Xs = X.iloc[-max_samples:]
        ys = y.iloc[-max_samples:]
    else:
        Xs, ys = X, y

    base = {
        "objective": config["model"]["objective"],
        "random_state": config["model"]["random_state"],
    }
    param_grid = tcfg["param_grid"]
    cv = TimeSeriesSplit(n_splits=int(tcfg["cv_splits"]))
    gs = GridSearchCV(
        xgb.XGBRegressor(**base),
        param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(Xs, ys)
    print(f"✓ Best tuning params: {gs.best_params_}  MAE={-gs.best_score_:.4f}")
    return dict(gs.best_params_)


def train_surge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Optional[Mapping[str, Any]] = None,
) -> xgb.XGBRegressor:
    """Train XGBoost with early stopping and print regression metrics on test."""
    cfg = dict(config or {})
    mp = cfg.get(
        "model",
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "objective": "reg:squarederror",
            "random_state": 42,
            "early_stopping_rounds": 20,
        },
    )

    es_rounds = int(mp.get("early_stopping_rounds", 20))
    xgb_kwargs: Dict[str, Any] = {
        "n_estimators": int(mp.get("n_estimators", 100)),
        "learning_rate": float(mp.get("learning_rate", 0.1)),
        "max_depth": int(mp.get("max_depth", 6)),
        "objective": str(mp.get("objective", "reg:squarederror")),
        "random_state": int(mp.get("random_state", 42)),
    }
    if es_rounds > 0:
        xgb_kwargs["early_stopping_rounds"] = es_rounds
    model = xgb.XGBRegressor(**xgb_kwargs)

    print(f"Training on {len(X_train):,} samples...")
    if es_rounds > 0:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    preds = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, preds)

    print("✓ Model training complete")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Features ({len(X_train.columns)}): {list(X_train.columns)}")

    # Stash metrics on the model for joblib consumers (API / metadata)
    model._surge_metrics_ = metrics  # type: ignore[attr-defined]
    return model
