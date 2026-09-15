from pathlib import Path

import pytest

from hireflow.evaluation.gemini import GeminiGroundedEvaluator, GroundedClaim, GroundedNarrative, _build_prompt
from hireflow.evaluation.heuristic import HeuristicGroundedEvaluator
from hireflow.exceptions import EvaluationError
from hireflow.models import CandidateMatch, CandidateProfile, EvidenceItem, JobRequirements, MatchComponentScores


def _profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="c1",
        source_file=Path("resume.pdf"),
        full_name="Jane Private",
        contact={"email": "jane@example.com", "phone": "+1-555-999"},
        headline="Senior Accountant",
        professional_summary="Experienced accountant focused on financial reporting.",
        skills=["Financial Reporting", "Tax Preparation"],
        software=["Microsoft Excel"],
        total_experience_years=4,
    )


def _job() -> JobRequirements:
    return JobRequirements(job_id="j1", title="Senior Accountant", required_skills=["Financial reporting", "GAAP"])


def _match() -> CandidateMatch:
    return CandidateMatch(
        candidate_id="c1",
        rank=1,
        retrieval_score=0.4,
        component_scores=MatchComponentScores(semantic=63, required_skills=80, experience=100, role=100, preferred=50),
        final_score=82,
        matched_requirements=["Financial reporting", "Microsoft Excel"],
        missing_requirements=["GAAP"],
        evidence=[
            EvidenceItem(evidence_id="E003", requirement="Financial reporting", evidence="Financial Reporting", source_section="technical_skills")
        ],
    )


def test_prompt_excludes_candidate_identity_and_contact():
    prompt, _ = _build_prompt(_profile(), _job(), _match())
    assert "Jane Private" not in prompt
    assert "jane@example.com" not in prompt
    assert "+1-555-999" not in prompt
    assert "Financial Reporting" in prompt


def test_heuristic_evaluator_preserves_deterministic_score():
    result = HeuristicGroundedEvaluator().evaluate(_profile(), _job(), _match())
    assert result.overall_score == 82
    assert result.recommendation.value == "recommended"
    assert "No direct resume evidence" in result.gaps[0]


class _FakeGeminiEvaluator(GeminiGroundedEvaluator):
    def __init__(self, response: GroundedNarrative):
        self.response = response
        self.api_key = "fake"
        self.model = "fake"
        self._client = None

    def _generate(self, prompt: str) -> GroundedNarrative:
        return self.response


def test_gemini_evidence_ids_are_resolved_server_side():
    # Evidence IDs are stable: E001 headline, E002 summary, E003 first skill.
    evaluator = _FakeGeminiEvaluator(
        GroundedNarrative(
            candidate_id="c1",
            strengths=[GroundedClaim(claim="Relevant financial reporting experience.", evidence_ids=["E003"])],
            gap_requirements=["GAAP"],
            reasoning="Strong alignment with a documented reporting skill; GAAP was not directly evidenced.",
            confidence=0.85,
        )
    )
    result = evaluator.evaluate(_profile(), _job(), _match())
    assert result.overall_score == 82  # LLM cannot replace deterministic score.
    assert result.evidence[0].evidence == "Financial Reporting"
    assert result.evidence[0].source_section == "technical_skills"


def test_gemini_rejects_unknown_evidence_ids():
    evaluator = _FakeGeminiEvaluator(
        GroundedNarrative(
            candidate_id="c1",
            strengths=[GroundedClaim(claim="Invented claim", evidence_ids=["E999"])],
            gap_requirements=["GAAP"],
            reasoning="x",
            confidence=0.5,
        )
    )
    with pytest.raises(EvaluationError, match="unknown evidence"):
        evaluator.evaluate(_profile(), _job(), _match())
