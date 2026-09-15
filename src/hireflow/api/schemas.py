"""Public API contracts that deliberately exclude resume raw text/contact data."""
from __future__ import annotations

from pydantic import Field

from hireflow.models import EvidenceItem, JobRequirements, MatchComponentScores, Recommendation
from hireflow.models.schemas import CandidateProfile, StrictModel
from hireflow.product import ProductSearchBundle, ProductSearchConfig


class StructuredSearchRequest(StrictModel):
    job: JobRequirements
    candidates: list[CandidateProfile] = Field(min_length=1)
    config: ProductSearchConfig = Field(default_factory=ProductSearchConfig)


class ApiCandidateResult(StrictModel):
    candidate_id: str
    rank: int
    headline: str | None = None
    total_experience_years: float | None = None
    overall_score: float
    recommendation: Recommendation
    component_scores: MatchComponentScores | None = None
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reasoning: str
    confidence: float


class SearchApiResponse(StrictModel):
    request_id: str
    job_id: str
    job_title: str
    candidate_count: int
    processing_time_ms: float
    embedding_model: str
    vector_backend: str
    evaluator_used: str
    cache_hit: bool
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    results: list[ApiCandidateResult] = Field(default_factory=list)


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str
    environment: str


class ReadyResponse(HealthResponse):
    checks: dict[str, str]


class ProblemDetail(StrictModel):
    error: str
    message: str
    request_id: str


def bundle_to_api_response(bundle: ProductSearchBundle, request_id: str) -> SearchApiResponse:
    """Convert a product bundle into a privacy-minimized API payload."""
    rows: list[ApiCandidateResult] = []
    for fallback_rank, item in enumerate(bundle.results, start=1):
        rows.append(
            ApiCandidateResult(
                candidate_id=item.profile.candidate_id,
                rank=item.match.rank or fallback_rank,
                headline=item.profile.headline,
                total_experience_years=item.profile.total_experience_years,
                overall_score=item.evaluation.overall_score,
                recommendation=item.evaluation.recommendation,
                component_scores=item.match.component_scores,
                strengths=item.evaluation.strengths,
                gaps=item.evaluation.gaps,
                matched_requirements=item.match.matched_requirements,
                missing_requirements=item.match.missing_requirements,
                evidence=item.evaluation.evidence,
                reasoning=item.evaluation.reasoning,
                confidence=item.evaluation.confidence,
            )
        )
    return SearchApiResponse(
        request_id=request_id,
        job_id=bundle.job.job_id,
        job_title=bundle.job.title,
        candidate_count=bundle.candidate_count,
        processing_time_ms=bundle.processing_time_ms,
        embedding_model=bundle.embedding_model,
        vector_backend=bundle.vector_backend,
        evaluator_used=bundle.evaluator_used,
        cache_hit=bundle.cache_hit,
        stage_timings_ms=bundle.stage_timings_ms,
        results=rows,
    )


__all__ = [
    "ApiCandidateResult",
    "HealthResponse",
    "ProblemDetail",
    "ReadyResponse",
    "SearchApiResponse",
    "StructuredSearchRequest",
    "bundle_to_api_response",
]
