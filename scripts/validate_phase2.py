#!/usr/bin/env python
"""Data-free Phase 2 smoke test suitable for CI."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.embeddings import TfidfEmbeddingModel  # noqa: E402
from hireflow.models import CandidateProfile, JobRequirements  # noqa: E402
from hireflow.retrieval import aggregate_candidate_hits, build_vector_index, candidate_documents  # noqa: E402


def main() -> None:
    candidates = [
        CandidateProfile(candidate_id="c1", source_file="c1.pdf", headline="Senior Accountant", skills=["GAAP", "Financial Reporting"], software=["Excel"]),
        CandidateProfile(candidate_id="c2", source_file="c2.pdf", headline="Software Engineer", skills=["Python", "Docker"]),
    ]
    job = JobRequirements(job_id="j1", title="Senior Accountant", required_skills=["GAAP", "Financial Reporting"], required_software=["Excel"])
    docs = [d for c in candidates for d in candidate_documents(c)]
    embedder = TfidfEmbeddingModel().fit([d.text for d in docs])
    vectors = embedder.embed_documents([d.text for d in docs])
    index = build_vector_index(vectors, docs, backend="numpy")
    results = aggregate_candidate_hits(index.search(embedder.embed_query(job.matching_text()), k=2), 2)
    assert results[0].candidate_id == "c1"
    print("HireFlow Phase 2 validation passed.")


if __name__ == "__main__":
    main()
