"""Shared interfaces and deterministic policy for candidate evaluation."""
from __future__ import annotations

from typing import Protocol

from hireflow.models import CandidateEvaluation, CandidateMatch, CandidateProfile, JobRequirements, Recommendation


class CandidateEvaluator(Protocol):
    def evaluate(
        self,
        profile: CandidateProfile,
        job: JobRequirements,
        match: CandidateMatch,
    ) -> CandidateEvaluation: ...


def recommendation_for_score(score: float) -> Recommendation:
    if score >= 85:
        return Recommendation.HIGHLY_RECOMMENDED
    if score >= 70:
        return Recommendation.RECOMMENDED
    if score >= 55:
        return Recommendation.CONSIDER
    return Recommendation.NOT_RECOMMENDED
