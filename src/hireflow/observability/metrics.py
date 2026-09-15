"""Prometheus metrics for API and ML pipeline observability."""
from __future__ import annotations

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "hireflow_http_requests_total",
    "HTTP requests handled by HireFlow",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "hireflow_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)
SEARCH_REQUESTS = Counter(
    "hireflow_search_requests_total",
    "Candidate search requests",
    ["embedding_provider", "evaluator", "cache"],
)
SEARCH_LATENCY = Histogram(
    "hireflow_search_duration_seconds",
    "End-to-end candidate search latency",
    ["embedding_provider", "evaluator"],
)
CANDIDATES_PROCESSED = Counter(
    "hireflow_candidates_processed_total",
    "Candidates processed by search",
)
PIPELINE_STAGE_LATENCY = Histogram(
    "hireflow_pipeline_stage_duration_seconds",
    "Internal ML pipeline stage latency",
    ["stage"],
)


__all__ = [
    "CANDIDATES_PROCESSED",
    "HTTP_LATENCY",
    "HTTP_REQUESTS",
    "PIPELINE_STAGE_LATENCY",
    "SEARCH_LATENCY",
    "SEARCH_REQUESTS",
]
