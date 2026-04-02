# Hugging Face Spaces: build React, serve API + static UI on port 7860
FROM node:20-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /src/frontend/dist ./frontend_dist
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
