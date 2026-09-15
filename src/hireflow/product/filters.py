"""Pure filtering helpers used by the recruiter results screen."""
from __future__ import annotations

from collections.abc import Iterable

from hireflow.models import Recommendation
from hireflow.product.models import CandidateProductResult


def filter_results(
    results: Iterable[CandidateProductResult],
    *,
    min_score: float = 0.0,
    recommendations: set[Recommendation] | None = None,
    role_query: str = "",
    min_experience: float | None = None,
    max_experience: float | None = None,
) -> list[CandidateProductResult]:
    query = role_query.casefold().strip()
    allowed = recommendations or set(Recommendation)
    filtered: list[CandidateProductResult] = []
    for item in results:
        score = float(item.match.final_score or 0.0)
        years = item.profile.total_experience_years
        title = item.profile.headline or ""
        if score < min_score:
            continue
        if item.evaluation.recommendation not in allowed:
            continue
        if query and query not in title.casefold():
            continue
        if min_experience is not None and (years is None or years < min_experience):
            continue
        if max_experience is not None and (years is None or years > max_experience):
            continue
        filtered.append(item)
    return filtered
