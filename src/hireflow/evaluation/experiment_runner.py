"""Experiment runner for retrieval and hybrid ranking benchmarks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hireflow.embeddings import GeminiEmbeddingModel, TfidfEmbeddingModel
from hireflow.evaluation.ranking_metrics import RankingMetrics, evaluate_ranking
from hireflow.models import CandidateProfile, JobRequirements
from hireflow.ranking import HybridCandidateReranker, MatchingTaxonomy, RequirementMatcher, ScoringWeights
from hireflow.retrieval import aggregate_candidate_hits, load_vector_index


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_name: str
    embedding_provider: str
    document_mode: str
    ranking_method: str
    ranked_candidate_ids: list[str]
    metrics: RankingMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "embedding_provider": self.embedding_provider,
            "document_mode": self.document_mode,
            "ranking_method": self.ranking_method,
            "ranked_candidate_ids": self.ranked_candidate_ids,
            "metrics": self.metrics.to_dict(),
        }


def _embedding_model(index_dir: Path, gemini_api_key: str | None = None):
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["embedding_provider"] == "tfidf":
        return TfidfEmbeddingModel.load(index_dir / "tfidf_vectorizer.joblib"), manifest
    if not gemini_api_key:
        raise ValueError("gemini_api_key is required for a Gemini embedding experiment")
    return GeminiEmbeddingModel(gemini_api_key, manifest["embedding_model"]), manifest


def retrieve_all_candidates(
    job: JobRequirements,
    index_dir: Path,
    candidate_count: int,
    gemini_api_key: str | None = None,
):
    embedding_model, manifest = _embedding_model(index_dir, gemini_api_key)
    index = load_vector_index(index_dir)
    query_vector = embedding_model.embed_query(job.matching_text())
    search_k = len(index.documents)
    hits = index.search(query_vector, k=search_k)
    results = aggregate_candidate_hits(hits, top_k_candidates=candidate_count)
    return results, manifest


def run_retrieval_experiment(
    experiment_name: str,
    job: JobRequirements,
    index_dir: Path,
    labels: dict[str, int],
    ks: tuple[int, ...],
    relevant_threshold: int,
    gemini_api_key: str | None = None,
) -> ExperimentResult:
    retrieved, manifest = retrieve_all_candidates(
        job, index_dir, candidate_count=len(labels), gemini_api_key=gemini_api_key
    )
    ranked_ids = [item.candidate_id for item in retrieved]
    metrics = evaluate_ranking(ranked_ids, labels, ks=ks, relevant_threshold=relevant_threshold)
    return ExperimentResult(
        experiment_name=experiment_name,
        embedding_provider=manifest["embedding_provider"],
        document_mode=manifest["document_mode"],
        ranking_method="semantic_retrieval",
        ranked_candidate_ids=ranked_ids,
        metrics=metrics,
    )


def run_hybrid_experiment(
    experiment_name: str,
    job: JobRequirements,
    candidates: dict[str, CandidateProfile],
    index_dir: Path,
    taxonomy_path: Path,
    labels: dict[str, int],
    weights: ScoringWeights,
    ks: tuple[int, ...],
    relevant_threshold: int,
    gemini_api_key: str | None = None,
) -> ExperimentResult:
    retrieved, manifest = retrieve_all_candidates(
        job, index_dir, candidate_count=len(labels), gemini_api_key=gemini_api_key
    )
    matcher = RequirementMatcher(MatchingTaxonomy.load(taxonomy_path))
    reranker = HybridCandidateReranker(matcher, weights)
    reranked = reranker.rerank(retrieved, candidates, job, top_k=None)
    ranked_ids = [item.candidate_id for item in reranked]
    metrics = evaluate_ranking(ranked_ids, labels, ks=ks, relevant_threshold=relevant_threshold)
    return ExperimentResult(
        experiment_name=experiment_name,
        embedding_provider=manifest["embedding_provider"],
        document_mode=manifest["document_mode"],
        ranking_method="hybrid_reranking",
        ranked_candidate_ids=ranked_ids,
        metrics=metrics,
    )
