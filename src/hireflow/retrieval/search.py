"""Candidate retrieval and chunk-to-candidate aggregation."""
from __future__ import annotations

from collections import defaultdict

from hireflow.models import CandidateRetrievalResult, RetrievalHit


def aggregate_candidate_hits(hits: list[RetrievalHit], top_k_candidates: int) -> list[CandidateRetrievalResult]:
    """Aggregate section-level hits into a candidate-level ranking.

    Score = 0.7 * best section score + 0.3 * mean of up to the best 3 section
    scores. For full-resume indexing there is one hit per candidate, so the
    formula reduces exactly to the original retrieval score.
    """
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.candidate_id].append(hit)

    results: list[CandidateRetrievalResult] = []
    for candidate_id, candidate_hits in grouped.items():
        ordered = sorted(candidate_hits, key=lambda item: item.score, reverse=True)
        top_scores = [item.score for item in ordered[:3]]
        best = top_scores[0]
        mean_top = sum(top_scores) / len(top_scores)
        score = 0.7 * best + 0.3 * mean_top
        results.append(
            CandidateRetrievalResult(
                candidate_id=candidate_id,
                retrieval_score=max(-1.0, min(1.0, score)),
                best_section=ordered[0].section,
                hits=ordered,
            )
        )

    return sorted(results, key=lambda item: item.retrieval_score, reverse=True)[:top_k_candidates]
