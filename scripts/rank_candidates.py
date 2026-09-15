#!/usr/bin/env python
"""Run semantic retrieval -> hybrid reranking -> grounded candidate evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.config import get_settings  # noqa: E402
from hireflow.embeddings import GeminiEmbeddingModel, TfidfEmbeddingModel  # noqa: E402
from hireflow.evaluation import EvaluationPipeline, GeminiGroundedEvaluator, HeuristicGroundedEvaluator  # noqa: E402
from hireflow.models import CandidateProfile, JobRequirements  # noqa: E402
from hireflow.ranking import HybridCandidateReranker, MatchingTaxonomy, RequirementMatcher, ScoringWeights  # noqa: E402
from hireflow.retrieval import aggregate_candidate_hits, load_vector_index  # noqa: E402
from hireflow.utils import read_json, read_jsonl  # noqa: E402


def run_pipeline(
    job_path: Path,
    candidates_path: Path,
    index_dir: Path,
    taxonomy_path: Path,
    top_k_retrieval: int,
    top_k_rerank: int,
    final_top_k: int,
    evaluator_name: str = "heuristic",
) -> dict[str, object]:
    settings = get_settings()
    job = read_json(job_path, JobRequirements)
    candidates = read_jsonl(candidates_path, CandidateProfile)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

    if manifest["embedding_provider"] == "tfidf":
        embedding_model = TfidfEmbeddingModel.load(index_dir / "tfidf_vectorizer.joblib")
    else:
        embedding_model = GeminiEmbeddingModel(settings.gemini_api_key or "", manifest["embedding_model"])

    index = load_vector_index(index_dir)
    query_vector = embedding_model.embed_query(job.matching_text())
    multiplier = 4 if manifest["document_mode"] == "section" else 1
    hits = index.search(query_vector, k=min(len(index.documents), top_k_retrieval * multiplier))
    retrieved = aggregate_candidate_hits(hits, top_k_candidates=top_k_retrieval)
    retrieval_rank = {item.candidate_id: idx for idx, item in enumerate(retrieved, start=1)}

    taxonomy = MatchingTaxonomy.load(taxonomy_path)
    matcher = RequirementMatcher(taxonomy)
    weights = ScoringWeights(
        semantic=settings.weight_semantic,
        required_skills=settings.weight_required_skills,
        experience=settings.weight_experience,
        role=settings.weight_role,
        preferred=settings.weight_preferred,
    )
    reranker = HybridCandidateReranker(matcher, weights)
    matches = reranker.rerank(retrieved, candidate_map, job, top_k=top_k_rerank)

    if evaluator_name == "gemini":
        evaluator = GeminiGroundedEvaluator(settings.gemini_api_key or "", settings.gemini_llm_model)
    elif evaluator_name == "heuristic":
        evaluator = HeuristicGroundedEvaluator()
    else:
        raise ValueError("evaluator must be 'heuristic' or 'gemini'")

    evaluations = EvaluationPipeline(evaluator).evaluate_shortlist(
        matches[:final_top_k], candidate_map, job
    )
    evaluation_map = {item.candidate_id: item for item in evaluations}

    rows: list[dict[str, object]] = []
    for match in matches:
        candidate = candidate_map[match.candidate_id]
        row: dict[str, object] = {
            "rerank_rank": match.rank,
            "retrieval_rank": retrieval_rank.get(match.candidate_id),
            "candidate_id": match.candidate_id,
            "headline": candidate.headline,
            "experience_years": candidate.total_experience_years,
            "retrieval_score": round(match.retrieval_score, 4),
            "final_score": match.final_score,
            "component_scores": match.component_scores.model_dump() if match.component_scores else None,
            "matched_requirements": match.matched_requirements,
            "missing_requirements": match.missing_requirements,
        }
        if match.candidate_id in evaluation_map:
            row["evaluation"] = evaluation_map[match.candidate_id].model_dump(mode="json")
        rows.append(row)

    return {
        "job_id": job.job_id,
        "job_title": job.title,
        "embedding_provider": manifest["embedding_provider"],
        "document_mode": manifest["document_mode"],
        "evaluator": evaluator_name,
        "weights": {
            "semantic": weights.semantic,
            "required_skills": weights.required_skills,
            "experience": weights.experience,
            "role": weights.role,
            "preferred": weights.preferred,
        },
        "results": rows,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, default=PROJECT_ROOT / "data" / "processed" / "job.json")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "data" / "processed" / "candidates.jsonl")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "indexes" / "baseline_full_tfidf")
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "matching_taxonomy.yaml")
    parser.add_argument("--top-k-retrieval", type=int, default=settings.top_k_retrieval)
    parser.add_argument("--top-k-rerank", type=int, default=settings.top_k_rerank)
    parser.add_argument("--final-top-k", type=int, default=settings.final_top_k)
    parser.add_argument("--evaluator", choices=["heuristic", "gemini"], default="heuristic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_pipeline(
        args.job,
        args.candidates,
        args.index_dir,
        args.taxonomy,
        args.top_k_retrieval,
        args.top_k_rerank,
        args.final_top_k,
        args.evaluator,
    )
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
