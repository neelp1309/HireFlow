"""Application service orchestrating the end-to-end HireFlow pipeline.

The Streamlit UI and FastAPI layer are intentionally thin. All ML orchestration,
cache semantics, latency instrumentation and privacy-safe operational logging live
here so every product surface executes the same tested pipeline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

from hireflow.config import Settings, get_settings
from hireflow.embeddings import GeminiEmbeddingModel, TfidfEmbeddingModel
from hireflow.evaluation import EvaluationPipeline, GeminiGroundedEvaluator, HeuristicGroundedEvaluator
from hireflow.exceptions import ConfigurationError
from hireflow.models import CandidateProfile, JobRequirements
from hireflow.observability.metrics import (
    CANDIDATES_PROCESSED,
    PIPELINE_STAGE_LATENCY,
    SEARCH_LATENCY,
    SEARCH_REQUESTS,
)
from hireflow.parsing.job_parser import parse_job_pdf
from hireflow.parsing.resume_parser import parse_resume_pdf
from hireflow.production.cache import SearchBundleCache, build_search_cache_key
from hireflow.ranking import HybridCandidateReranker, MatchingTaxonomy, RequirementMatcher, ScoringWeights
from hireflow.retrieval import aggregate_candidate_hits, build_vector_index, candidate_documents

from hireflow.product.models import CandidateProductResult, ProductSearchBundle, ProductSearchConfig

logger = logging.getLogger(__name__)


class _StageClock:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def run(self, stage: str, fn):
        started = time.perf_counter()
        value = fn()
        elapsed = (time.perf_counter() - started) * 1000.0
        self.timings[stage] = round(elapsed, 2)
        PIPELINE_STAGE_LATENCY.labels(stage=stage).observe(elapsed / 1000.0)
        return value


class HireFlowProductService:
    """Reusable product-facing facade around parsing, retrieval and evaluation."""

    def __init__(
        self,
        taxonomy_path: str | Path | None = None,
        settings: Settings | None = None,
        search_cache: SearchBundleCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        default_taxonomy = Path(__file__).resolve().parents[3] / "configs" / "matching_taxonomy.yaml"
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else default_taxonomy
        taxonomy = MatchingTaxonomy.load(self.taxonomy_path)
        self.matcher = RequirementMatcher(taxonomy)
        if search_cache is not None:
            self.search_cache = search_cache
        elif self.settings.search_cache_enabled:
            self.search_cache = SearchBundleCache(
                max_entries=self.settings.search_cache_max_entries,
                ttl_seconds=self.settings.search_cache_ttl_seconds,
            )
        else:
            self.search_cache = None

    def _weights(self) -> ScoringWeights:
        return ScoringWeights(
            semantic=self.settings.weight_semantic,
            required_skills=self.settings.weight_required_skills,
            experience=self.settings.weight_experience,
            role=self.settings.weight_role,
            preferred=self.settings.weight_preferred,
        )

    def _weights_dict(self) -> dict[str, float]:
        return {
            "semantic": self.settings.weight_semantic,
            "required_skills": self.settings.weight_required_skills,
            "experience": self.settings.weight_experience,
            "role": self.settings.weight_role,
            "preferred": self.settings.weight_preferred,
        }

    def _embedding_model(self, provider: str, texts: list[str]):
        if provider == "gemini":
            if not self.settings.gemini_api_key:
                raise ConfigurationError(
                    "Gemini embeddings were selected but GEMINI_API_KEY is not configured. "
                    "Use TF-IDF for offline mode or add the key to .env."
                )
            model = GeminiEmbeddingModel(
                self.settings.gemini_api_key,
                self.settings.gemini_embedding_model,
                max_retries=self.settings.gemini_max_retries,
                initial_backoff_seconds=self.settings.gemini_initial_backoff_seconds,
            )
            return model.fit(texts)
        return TfidfEmbeddingModel().fit(texts)

    def _evaluator(self, requested: str):
        if requested == "gemini" and self.settings.gemini_api_key:
            return GeminiGroundedEvaluator(
                self.settings.gemini_api_key,
                self.settings.gemini_llm_model,
                max_retries=self.settings.gemini_max_retries,
                initial_backoff_seconds=self.settings.gemini_initial_backoff_seconds,
            ), "gemini"
        if requested == "gemini":
            return HeuristicGroundedEvaluator(), "heuristic (Gemini key unavailable)"
        return HeuristicGroundedEvaluator(), "heuristic"

    def _validate_candidates(self, candidates: Iterable[CandidateProfile]) -> list[CandidateProfile]:
        ordered = list(candidates)
        if not ordered:
            raise ValueError("At least one candidate profile is required")
        if len(ordered) > self.settings.max_candidates_per_request:
            raise ValueError(
                f"At most {self.settings.max_candidates_per_request} candidates are allowed per request"
            )
        seen: set[str] = set()
        duplicates: set[str] = set()
        for profile in ordered:
            if profile.candidate_id in seen:
                duplicates.add(profile.candidate_id)
            seen.add(profile.candidate_id)
        if duplicates:
            raise ValueError(f"Duplicate candidate IDs are not allowed: {sorted(duplicates)}")
        return ordered

    def search_structured(
        self,
        job: JobRequirements,
        candidates: Iterable[CandidateProfile],
        config: ProductSearchConfig | None = None,
    ) -> ProductSearchBundle:
        """Run a production-instrumented search from validated structures."""
        request_started = time.perf_counter()
        cfg = config or ProductSearchConfig()
        candidate_list = self._validate_candidates(candidates)
        candidate_map = {item.candidate_id: item for item in candidate_list}
        cache_key = build_search_cache_key(job, candidate_list, cfg, self._weights_dict())

        if self.search_cache is not None:
            cached = self.search_cache.get(cache_key)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                SEARCH_REQUESTS.labels(
                    embedding_provider=cfg.embedding_provider,
                    evaluator=cfg.evaluator,
                    cache="hit",
                ).inc()
                SEARCH_LATENCY.labels(
                    embedding_provider=cfg.embedding_provider,
                    evaluator=cfg.evaluator,
                ).observe(elapsed_ms / 1000.0)
                logger.info(
                    "candidate_search_cache_hit",
                    extra={
                        "job_id": job.job_id,
                        "candidate_count": len(candidate_list),
                        "elapsed_ms": round(elapsed_ms, 2),
                    },
                )
                return cached.model_copy(
                    update={
                        "cache_hit": True,
                        "processing_time_ms": round(elapsed_ms, 2),
                        "stage_timings_ms": {"cache_lookup": round(elapsed_ms, 2)},
                    }
                )

        clock = _StageClock()
        documents = clock.run(
            "document_build",
            lambda: [
                document
                for candidate in candidate_list
                for document in candidate_documents(candidate, mode=cfg.document_mode)
            ],
        )
        texts = [document.text for document in documents]
        model = clock.run("embedding_fit", lambda: self._embedding_model(cfg.embedding_provider, texts))
        vectors = clock.run("document_embedding", lambda: model.embed_documents(texts))
        index = clock.run(
            "vector_index_build",
            lambda: build_vector_index(vectors, documents, backend=cfg.vector_backend),
        )

        query_vector = clock.run("query_embedding", lambda: model.embed_query(job.matching_text()))
        hit_multiplier = 4 if cfg.document_mode == "section" else 1
        hit_k = min(len(documents), max(cfg.top_k_retrieval, cfg.top_k_retrieval * hit_multiplier))
        hits = clock.run("vector_search", lambda: index.search(query_vector, hit_k))
        retrieval = clock.run(
            "candidate_aggregation",
            lambda: aggregate_candidate_hits(hits, min(cfg.top_k_retrieval, len(candidate_list))),
        )

        reranker = HybridCandidateReranker(self.matcher, self._weights())
        matches = clock.run(
            "hybrid_rerank",
            lambda: reranker.rerank(
                retrieval,
                candidate_map,
                job,
                top_k=min(cfg.top_k_rerank, len(retrieval)),
            ),
        )
        shortlist = matches[: min(cfg.final_top_k, len(matches))]

        evaluator, evaluator_used = self._evaluator(cfg.evaluator)
        evaluation_pipeline = EvaluationPipeline(evaluator)
        evaluations = clock.run(
            "grounded_evaluation",
            lambda: evaluation_pipeline.evaluate_shortlist(shortlist, candidate_map, job),
        )
        evaluation_by_id = {item.candidate_id: item for item in evaluations}
        results = [
            CandidateProductResult(
                profile=candidate_map[match.candidate_id],
                match=match,
                evaluation=evaluation_by_id[match.candidate_id],
            )
            for match in shortlist
        ]
        elapsed_ms = (time.perf_counter() - request_started) * 1000.0
        bundle = ProductSearchBundle(
            job=job,
            results=results,
            candidate_count=len(candidate_list),
            processing_time_ms=round(elapsed_ms, 2),
            embedding_model=model.model_name,
            vector_backend=index.backend_name,
            evaluator_used=evaluator_used,
            config=cfg,
            cache_hit=False,
            stage_timings_ms=clock.timings,
        )
        if self.search_cache is not None:
            self.search_cache.set(cache_key, bundle)

        SEARCH_REQUESTS.labels(
            embedding_provider=cfg.embedding_provider,
            evaluator=cfg.evaluator,
            cache="miss",
        ).inc()
        SEARCH_LATENCY.labels(
            embedding_provider=cfg.embedding_provider,
            evaluator=cfg.evaluator,
        ).observe(elapsed_ms / 1000.0)
        CANDIDATES_PROCESSED.inc(len(candidate_list))
        logger.info(
            "candidate_search_completed",
            extra={
                "job_id": job.job_id,
                "candidate_count": len(candidate_list),
                "shortlist_count": len(results),
                "embedding_provider": cfg.embedding_provider,
                "vector_backend": index.backend_name,
                "evaluator": evaluator_used,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        return bundle

    def search_paths(
        self,
        job_pdf: str | Path,
        resume_pdfs: Iterable[str | Path],
        config: ProductSearchConfig | None = None,
    ) -> ProductSearchBundle:
        """Parse local PDFs and run the complete product pipeline."""
        parse_started = time.perf_counter()
        job = parse_job_pdf(job_pdf)
        paths = sorted((Path(path) for path in resume_pdfs), key=lambda item: item.name.casefold())
        candidates = [parse_resume_pdf(path) for path in paths]
        parse_ms = round((time.perf_counter() - parse_started) * 1000.0, 2)
        PIPELINE_STAGE_LATENCY.labels(stage="pdf_parsing").observe(parse_ms / 1000.0)
        bundle = self.search_structured(job, candidates, config)
        timings = dict(bundle.stage_timings_ms)
        timings["pdf_parsing"] = parse_ms
        return bundle.model_copy(
            update={
                "stage_timings_ms": timings,
                "processing_time_ms": round(bundle.processing_time_ms + parse_ms, 2),
            }
        )
