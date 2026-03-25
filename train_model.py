"""
Train and save XGBoost model for deployment.
Produces a .pkl file and model_info.pkl for the API (FastAPI).
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
import math

import joblib
import pandas as pd

from config_loader import load_config
from modeling import (
    prepare_cyclical_features,
    time_series_cv_scores,
    train_surge_model,
    tune_hyperparameters,
)
from weather_service import fetch_nyc_weather

TARGET = "Target_DER_t+15"

# Columns to exclude from model inputs (identifiers, targets, raw counts)
EXCLUDE_FROM_X = {
    "Time_Bin",
    "Target_DER_t+15",
    "DER_t",
    "ActiveRequests",
    "AvailableDriversProxy",
    "hour",  # use cyclical encoding only
}


def _maybe_mlflow_log(config: dict, params: dict, metrics: dict, model, feature_names: list) -> None:
    if not config.get("mlflow", {}).get("enabled"):
        return
    try:
        import mlflow
    except ImportError:
        print("MLflow enabled in config but package not installed; skipping tracking.")
        return
    exp = config["mlflow"].get("experiment_name", "surge_prediction")
    mlflow.set_experiment(exp)
    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isfinite(v):
                continue
            mlflow.log_metric(k, float(v))
        mlflow.set_tag("n_features", str(len(feature_names)))
        try:
            mlflow.xgboost.log_model(model, artifact_path="model")
        except Exception:
            mlflow.sklearn.log_model(model, artifact_path="model")


def train_and_save() -> object:
    """Train model, save versioned artifact + latest symlink path for API."""
    print("=" * 60)
    print("TRAINING MODEL FOR DEPLOYMENT")
    print("=" * 60)

    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        raise

    processed_dir = config["paths"]["processed_dir"]
    model_dir = config["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load processed data
    print("\n[1/6] Loading processed taxi data...")
    try:
        df_train = pd.read_parquet(f"{processed_dir}/train_data.parquet")
        df_test = pd.read_parquet(f"{processed_dir}/test_data.parquet")
    except FileNotFoundError as e:
        print(f"Missing parquet data: {e}")
        print("Run retrieval.py first to build train/test parquet files.")
        raise
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}")

    # 2. Weather
    print("\n[2/6] Fetching weather data...")
    try:
        w_start = config["data"]["weather_start"]
        w_end = config["data"]["weather_end"]
        df_weather = fetch_nyc_weather(w_start, w_end)
        df_train = pd.merge(df_train, df_weather, on="Time_Bin", how="left")
        df_test = pd.merge(df_test, df_weather, on="Time_Bin", how="left")
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        traceback.print_exc()
        raise

    # 3. Time features
    print("\n[3/6] Engineering time features...")
    df_train = prepare_cyclical_features(df_train)
    df_test = prepare_cyclical_features(df_test)
    df_train["day_of_week"] = df_train["Time_Bin"].dt.dayofweek
    df_test["day_of_week"] = df_test["Time_Bin"].dt.dayofweek
    df_train["is_weekend"] = (df_train["day_of_week"] >= 5).astype(int)
    df_test["is_weekend"] = (df_test["day_of_week"] >= 5).astype(int)

    # 4. Feature list
    numeric_cols = df_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_FROM_X]
    feature_cols = sorted(set(feature_cols))

    df_train = df_train.dropna(subset=feature_cols + [TARGET])
    df_test = df_test.dropna(subset=feature_cols + [TARGET])

    X_train = df_train[feature_cols]
    y_train = df_train[TARGET]
    X_test = df_test[feature_cols]
    y_test = df_test[TARGET]

    print(f"  Final shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    # 5. Optional CV + tuning
    print("\n[4/6] Optional CV / hyperparameter search...")
    model_cfg = dict(config)
    base_params = {
        "n_estimators": int(config["model"]["n_estimators"]),
        "learning_rate": float(config["model"]["learning_rate"]),
        "max_depth": int(config["model"]["max_depth"]),
        "objective": str(config["model"]["objective"]),
        "random_state": int(config["model"]["random_state"]),
    }
    try:
        n_splits = min(5, max(2, len(X_train) // 100))
        cv_metrics = time_series_cv_scores(X_train, y_train, base_params, n_splits=n_splits)
        print(
            f"  TimeSeriesSplit MAE: {cv_metrics['cv_mae_mean']:.4f} "
            f"(±{cv_metrics['cv_mae_std']:.4f})"
        )
    except Exception as e:
        print(f"  (Skipping CV: {e})")
        cv_metrics = {"cv_mae_mean": float("nan"), "cv_mae_std": float("nan")}

    if config["tuning"]["enabled"]:
        best = tune_hyperparameters(X_train, y_train, config)
        config["model"].update(best)

    # 6. Train + save
    print("\n[5/6] Training XGBoost model...")
    model = train_surge_model(X_train, y_train, X_test, y_test, config=model_cfg)

    metrics = getattr(model, "_surge_metrics_", {})
    metrics = {**metrics, **cv_metrics}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(model_dir, f"xgboost_surge_model_{stamp}.pkl")
    latest_path = os.path.join(model_dir, "xgboost_surge_model.pkl")

    print("\n[6/6] Saving model...")
    try:
        joblib.dump(model, versioned_path)
        joblib.dump(model, latest_path)
        print(f"  ✓ Versioned model: {versioned_path}")
        print(f"  ✓ Latest copy:       {latest_path}")
    except OSError as e:
        print(f"Failed to save model: {e}")
        raise

    feature_info = {
        "features": feature_cols,
        "target": TARGET,
        "model_type": "XGBoost",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "artifact_version": versioned_path,
        "trained_at_utc": stamp,
    }
    joblib.dump(feature_info, os.path.join(model_dir, "model_info.pkl"))

    _maybe_mlflow_log(
        config,
        params=config["model"],
        metrics=metrics,
        model=model,
        feature_names=feature_cols,
    )

    print("\n" + "=" * 60)
    print("✓ MODEL READY FOR DEPLOYMENT")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start API: uvicorn api:app --host 0.0.0.0 --port 8000")
    print("  2. Test: curl http://localhost:8000/health")
    print("  3. Frontend: cd frontend && npm run dev")
    print("=" * 60)

    return model


if __name__ == "__main__":
    try:
        train_and_save()
    except Exception:
        sys.exit(1)
