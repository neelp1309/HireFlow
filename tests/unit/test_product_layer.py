from pathlib import Path

import pytest

from hireflow.models import CandidateEvaluation, CandidateMatch, CandidateProfile, MatchComponentScores, Recommendation
from hireflow.product import CandidateProductResult, ProductSearchConfig, filter_results


def _result(score: float, role: str, years: float, recommendation: Recommendation) -> CandidateProductResult:
    profile = CandidateProfile(
        candidate_id=f"c{int(score)}",
        source_file=Path("resume.pdf"),
        full_name="Example Candidate",
        headline=role,
        total_experience_years=years,
    )
    match = CandidateMatch(
        candidate_id=profile.candidate_id,
        rank=1,
        retrieval_score=0.5,
        final_score=score,
        component_scores=MatchComponentScores(
            semantic=60,
            required_skills=70,
            experience=80,
            role=90,
            preferred=50,
        ),
    )
    evaluation = CandidateEvaluation(
        candidate_id=profile.candidate_id,
        overall_score=score,
        recommendation=recommendation,
        reasoning="Grounded test result",
        confidence=0.8,
    )
    return CandidateProductResult(profile=profile, match=match, evaluation=evaluation)


def test_product_config_rejects_invalid_stage_sizes() -> None:
    with pytest.raises(ValueError):
        ProductSearchConfig(top_k_retrieval=5, top_k_rerank=10, final_top_k=5)


def test_product_filter_combines_score_role_experience_and_recommendation() -> None:
    results = [
        _result(82, "Senior Accountant", 5, Recommendation.RECOMMENDED),
        _result(68, "Financial Analyst", 3, Recommendation.CONSIDER),
    ]
    filtered = filter_results(
        results,
        min_score=75,
        recommendations={Recommendation.RECOMMENDED},
        role_query="accountant",
        min_experience=4,
        max_experience=6,
    )
    assert len(filtered) == 1
    assert filtered[0].profile.headline == "Senior Accountant"
