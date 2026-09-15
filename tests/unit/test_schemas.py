from pathlib import Path

import pytest
from pydantic import ValidationError

from hireflow.models.schemas import (
    CandidateEvaluation,
    CandidateProfile,
    JobRequirements,
    Recommendation,
)


def test_candidate_matching_text_excludes_identity() -> None:
    candidate = CandidateProfile(
        candidate_id="C001",
        source_file=Path("resume.pdf"),
        full_name="Private Candidate Name",
        headline="Senior Accountant",
        skills=["GAAP", "Excel", "gaap"],
        software=["QuickBooks"],
        total_experience_years=4,
    )

    text = candidate.matching_text()
    assert "Private Candidate Name" not in text
    assert "C001" not in text
    assert "Senior Accountant" in text
    assert candidate.skills == ["GAAP", "Excel"]


def test_job_matching_text_uses_relevant_requirements() -> None:
    job = JobRequirements(
        job_id="J001",
        title="Senior Accountant",
        experience_min_years=3,
        experience_max_years=5,
        required_skills=["Financial Reporting", "GAAP"],
        preferred_certifications=["CPA"],
    )
    text = job.matching_text()
    assert "Senior Accountant" in text
    assert "Financial Reporting" in text
    assert "CPA" in text


def test_evaluation_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        CandidateEvaluation(
            candidate_id="C001",
            overall_score=101,
            recommendation=Recommendation.HIGHLY_RECOMMENDED,
            reasoning="Invalid score should fail validation.",
            confidence=0.9,
        )
