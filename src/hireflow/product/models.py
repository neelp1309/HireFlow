"""Product-layer request/response contracts for the recruiter application."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from hireflow.models import CandidateEvaluation, CandidateMatch, CandidateProfile, JobRequirements
from hireflow.models.schemas import StrictModel


class ProductSearchConfig(StrictModel):
    """Runtime knobs exposed by the Streamlit product.

    The defaults mirror the experimentally evaluated Phase 4 baseline. More
    advanced production tuning belongs in Phase 6 rather than in UI code.
    """

    document_mode: Literal["full", "section"] = "full"
    embedding_provider: Literal["tfidf", "gemini"] = "tfidf"
    vector_backend: Literal["auto", "faiss", "numpy"] = "auto"
    evaluator: Literal["heuristic", "gemini"] = "heuristic"
    top_k_retrieval: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=10, ge=1, le=50)
    final_top_k: int = Field(default=5, ge=1, le=25)

    @model_validator(mode="after")
    def validate_stage_sizes(self) -> "ProductSearchConfig":
        if self.top_k_rerank > self.top_k_retrieval:
            raise ValueError("top_k_rerank cannot exceed top_k_retrieval")
        if self.final_top_k > self.top_k_rerank:
            raise ValueError("final_top_k cannot exceed top_k_rerank")
        return self


class CandidateProductResult(StrictModel):
    """One ranked candidate plus the explanation shown to recruiters."""

    profile: CandidateProfile
    match: CandidateMatch
    evaluation: CandidateEvaluation


class ProductSearchBundle(StrictModel):
    """Complete output of one recruiter search session."""

    job: JobRequirements
    results: list[CandidateProductResult]
    candidate_count: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0)
    embedding_model: str
    vector_backend: str
    evaluator_used: str
    config: ProductSearchConfig
    cache_hit: bool = False
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
