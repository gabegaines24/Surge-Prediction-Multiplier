# Backend (Training + API)

This directory contains the Python training pipeline and the FastAPI service.

## Prerequisites

From the project root, install dependencies:

```bash
pip install -e .
```

## Train & save the model

This runs `backend/train_model.py` as a module and writes:

- `models/xgboost_surge_model.pkl` (latest copy)
- `models/xgboost_surge_model_<UTC>.pkl` (versioned copy)
- `models/model_info.pkl` (feature order + metrics)

```bash
python -m backend.train_model
```

## Process taxi data (Dask pipeline)

```bash
python -m backend.retrieval
```

## Run the API

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict`
- `GET /model-info`

