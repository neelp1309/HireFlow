#!/usr/bin/env python
"""Search the resume index for candidates matching a processed JD."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.config import get_settings  # noqa: E402
from hireflow.embeddings import GeminiEmbeddingModel, TfidfEmbeddingModel  # noqa: E402
from hireflow.models import CandidateProfile, JobRequirements  # noqa: E402
from hireflow.retrieval import aggregate_candidate_hits, load_vector_index  # noqa: E402
from hireflow.utils import read_json, read_jsonl  # noqa: E402


def search(job_path: Path, candidates_path: Path, index_dir: Path, top_k: int) -> list[dict[str, object]]:
    job = read_json(job_path, JobRequirements)
    candidates = read_jsonl(candidates_path, CandidateProfile)
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

    if manifest["embedding_provider"] == "tfidf":
        model = TfidfEmbeddingModel.load(index_dir / "tfidf_vectorizer.joblib")
    else:
        settings = get_settings()
        model = GeminiEmbeddingModel(settings.gemini_api_key or "", manifest["embedding_model"])

    index = load_vector_index(index_dir)
    query_vector = model.embed_query(job.matching_text())
    # Section indexes need extra chunk recall before candidate aggregation.
    multiplier = 4 if manifest["document_mode"] == "section" else 1
    hits = index.search(query_vector, k=min(len(index.documents), top_k * multiplier))
    results = aggregate_candidate_hits(hits, top_k_candidates=top_k)

    output: list[dict[str, object]] = []
    for rank, result in enumerate(results, start=1):
        candidate = candidate_map[result.candidate_id]
        output.append(
            {
                "rank": rank,
                "candidate_id": result.candidate_id,
                "headline": candidate.headline,
                "experience_years": candidate.total_experience_years,
                "retrieval_score": round(result.retrieval_score, 4),
                "best_section": result.best_section,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, default=PROJECT_ROOT / "data" / "processed" / "job.json")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "data" / "processed" / "candidates.jsonl")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "indexes" / "baseline")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = search(args.job, args.candidates, args.index_dir, args.top_k)
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
