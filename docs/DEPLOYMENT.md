# Deployment Guide

## Local development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env
make test
```

## Streamlit

```bash
make run
```

Default URL: `http://localhost:8501`

## FastAPI

```bash
make api
```

Default URL: `http://localhost:8000`

## Docker

```bash
docker build -t hireflow:1.0.0 .
docker run --rm -p 8000:8000 --env-file .env hireflow:1.0.0
```

## Docker Compose

```bash
# API
docker compose up --build

# API + Streamlit
docker compose --profile ui up --build
```

## Optional FAISS backend

The default installation is cross-platform and can use exact NumPy cosine search. On environments with FAISS support:

```bash
pip install -r requirements-faiss.txt
```

Then set `vector_backend=faiss` or leave it as `auto`.

## Production configuration

Recommended environment variables:

```text
APP_ENV=production
LOG_LEVEL=INFO
API_ACCESS_KEY=<secret>
API_WORKERS=1
SEARCH_CACHE_ENABLED=true
SEARCH_CACHE_TTL_SECONDS=900
METRICS_ENABLED=true
```

If using Gemini:

```text
GEMINI_API_KEY=<secret>
GEMINI_MAX_RETRIES=3
GEMINI_INITIAL_BACKOFF_SECONDS=1.0
```

## Scaling considerations

The included in-memory cache and local vector index are suitable for a portfolio/reference deployment. For horizontal scaling, externalize:

- candidate/document storage;
- vector search;
- distributed cache;
- metrics aggregation;
- secrets;
- audit/event storage.

Large corpus ingestion should move to asynchronous workers rather than request-time parsing.
