# Phase 6 — Production Engineering

Phase 6 turns HireFlow from a local ML product into a deployable service with a reusable API,
bounded caching, observability, persistent corpus indexing, runtime hardening, containerization
and CI.

## Serving architecture

```text
Recruiter / Client
       |
       +--------------------+
       |                    |
  Streamlit UI          FastAPI API
       |                    |
       +---------+----------+
                 |
        HireFlowProductService
                 |
     +-----------+-----------+
     |           |           |
 Retrieval   Hybrid rank   Evaluation
     |                       |
  FAISS/NumPy          Heuristic/Gemini
```

Both UI and API call the same service. This avoids duplicated ranking logic and ensures the
same validation, cache behavior and deterministic scoring are applied everywhere.

## API endpoints

- `GET /health/live` — process liveness.
- `GET /health/ready` — checks service initialization, taxonomy availability and artifact writes.
- `GET /metrics` — Prometheus exposition format.
- `POST /v1/search/structured` — JSON `JobRequirements` + `CandidateProfile` search.
- `POST /v1/search/files` — multipart job PDF + resume PDFs or resume ZIP.

Search routes can be protected with `API_ACCESS_KEY`; clients send it as `X-API-Key`.
API responses intentionally omit resume raw text, contact details and candidate full names.
The Streamlit recruiter UI can still display identity locally, but identity remains excluded from
retrieval, ranking and LLM prompts.

## Search cache

`HireFlowProductService` contains a bounded process-local TTL cache. Cache keys use only:

- job matching text,
- candidate IDs and qualification matching text,
- parser version,
- an opaque SHA-256 identity-metadata invalidation digest,
- runtime search configuration,
- scoring weights.

Identity never contributes to ranking; the digest only prevents a recruiter-facing cached bundle
from being reused if display metadata changes for the same candidate ID. Cached `ProductSearchBundle` objects remain in memory;
they are not serialized to disk. This improves repeat-request latency without creating a new
persistent store of candidate PII.

## Incremental corpus indexing

`CorpusIndexManager` maintains private index artifacts under `artifacts/indexes/`.

For TF-IDF, the vocabulary depends on the complete corpus, so any document change triggers a
full refit. An unchanged corpus loads the persisted model and vector index with zero re-embedding.

For Gemini embeddings, the embedding space is corpus-independent. The manager fingerprints each
retrieval document and re-embeds only new/changed documents, reuses unchanged vectors, removes
deleted documents and then cheaply rebuilds the search index over the combined matrix.

This distinction is important: claiming TF-IDF supports true append-only embeddings would be
incorrect because adding documents can alter IDF weights.

## Observability

The API and product service expose:

- request IDs via `X-Request-ID`,
- structured JSON logs,
- HTTP request count and latency,
- search request count and latency,
- candidates processed,
- per-pipeline-stage latency,
- cache hit/miss labels.

Operational logs record job/candidate counts and IDs needed for debugging but never log resume
raw text, contact details or candidate names.

## Resilience

Gemini embedding and evaluation calls use bounded exponential-backoff retries. The retry count
and initial backoff are environment-configurable. The deterministic heuristic evaluator remains
available for offline operation and CI.

## Containers

The Docker image:

- uses Python 3.12 slim,
- installs dependencies into a build-stage virtual environment,
- runs as a non-root `hireflow` user,
- includes an HTTP liveness healthcheck,
- does not bake `.env`, raw resumes or runtime artifacts into the image.

`docker-compose.yml` starts the API by default; `docker compose --profile ui up --build` also
starts Streamlit.

## CI

GitHub Actions runs source compilation, the full unit/integration test suite, coverage, critical Ruff static checks and
a production Docker build on pull requests. A separate dependency review workflow checks newly
introduced dependencies.

## Production limitations

The current project is designed as a high-quality portfolio/reference implementation, not a
fully managed recruiting SaaS. A large-scale deployment should externalize the vector database,
use distributed caching, encrypt candidate stores, add user/tenant authorization, centralized
secret management, audit retention policies, rate limiting and asynchronous workers for large
batch ingestion.

## Supplied-dataset runtime validation

The Phase 6 build was exercised against the supplied Senior Accountant JD and all 50 usable
resume PDFs using the offline TF-IDF + NumPy + heuristic configuration. One cold file-based run
completed in approximately 576 ms in the build environment. Repeating the same request produced
a search-cache hit and completed in approximately 299 ms overall; about 297 ms of that warm run
was PDF parsing, while the cached post-parsing search lookup itself was approximately 1.3 ms.
These timings are environment-specific smoke measurements, not production SLAs.

The persistent TF-IDF corpus sync embedded all 50 documents on first build, then reused all 50
on an unchanged second sync with zero re-embedding. Because TF-IDF must be refit when the corpus
changes, changed-corpus incremental vector reuse is intentionally reserved for corpus-independent
dense embeddings.

The FastAPI file endpoint was also exercised with real supplied PDFs and returned ranked results
without exposing `raw_text`, contact fields or full names in the public response.

## Concurrency notes

FastAPI search endpoints execute the synchronous CPU/SDK pipeline in Starlette's worker threadpool
instead of blocking the event loop. The in-memory cache is thread-safe. Multiple Uvicorn worker
processes each maintain their own cache and Prometheus process state; a horizontally scaled
production deployment should use an external cache and a metrics architecture appropriate for
multi-process/container aggregation.
