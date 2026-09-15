from pathlib import Path

from hireflow.models import (
    CandidateProfile,
    CandidateRetrievalResult,
    Certification,
    Education,
    JobRequirements,
    RetrievalHit,
)
from hireflow.ranking import (
    HybridCandidateReranker,
    MatchingTaxonomy,
    RequirementMatcher,
    experience_alignment,
    role_alignment,
)


def _taxonomy() -> MatchingTaxonomy:
    return MatchingTaxonomy.load(Path(__file__).resolve().parents[2] / "configs" / "matching_taxonomy.yaml")


def _job() -> JobRequirements:
    return JobRequirements(
        job_id="j1",
        title="Senior Accountant",
        experience_min_years=3,
        experience_max_years=5,
        responsibilities=[
            "Prepare financial statements",
            "Manage month-end closing",
            "Prepare journal entries and account reconciliations",
            "Prepare tax returns",
        ],
        required_skills=["Strong knowledge of GAAP and financial reporting standards"],
        required_software=["Microsoft Excel", "Accounting software (QuickBooks, SAP, Oracle, or similar)"],
        education_requirements=["Bachelor's degree in Accounting, Finance, or related field"],
        preferred_qualifications=["Experience with financial modeling and forecasting"],
        preferred_certifications=["CPA"],
    )


def _candidate() -> CandidateProfile:
    return CandidateProfile(
        candidate_id="c1",
        source_file=Path("resume.pdf"),
        full_name="Private Person",
        headline="Senior Accountant",
        contact={"email": "private@example.com"},
        professional_summary="Accounting professional with accurate financial reporting experience.",
        skills=["Financial Reporting", "Month-End Closing", "Journal Entries", "Bank Reconciliation", "Tax Preparation"],
        software=["Microsoft Excel", "QuickBooks"],
        total_experience_years=4,
        education=[Education(degree="Bachelor of Science", field_of_study="Accounting")],
        certifications=[Certification(name="CPA")],
    )


def test_experience_alignment_penalizes_underqualification_more_than_overqualification():
    assert experience_alignment(4, 3, 5) == 100
    assert experience_alignment(2, 3, 5) == 55
    assert experience_alignment(6, 3, 5) == 88


def test_role_alignment_prefers_exact_accounting_role():
    exact = role_alignment("Senior Accountant", "Senior Accountant")
    nearby = role_alignment("Staff Accountant", "Senior Accountant")
    specialist = role_alignment("Tax Manager", "Senior Accountant")
    unrelated = role_alignment("VP", "Senior Accountant")
    assert exact == 100
    assert exact > nearby > specialist > unrelated


def test_hybrid_reranker_builds_explainable_match():
    candidate = _candidate()
    retrieval = CandidateRetrievalResult(
        candidate_id="c1",
        retrieval_score=0.36,
        best_section="full_resume",
        hits=[RetrievalHit(vector_id="c1:full", candidate_id="c1", section="full_resume", score=0.36, text="x")],
    )
    reranker = HybridCandidateReranker(RequirementMatcher(_taxonomy()))
    match = reranker.score_candidate(retrieval, candidate, _job())
    assert match.final_score is not None and match.final_score > 70
    assert match.component_scores is not None
    assert match.component_scores.role == 100
    assert "Microsoft Excel" in match.matched_requirements
    assert any(item.evidence_id for item in match.evidence)
    # PII remains metadata and never appears in evidence.
    evidence_text = " ".join(item.evidence for item in match.evidence)
    assert "Private Person" not in evidence_text
    assert "private@example.com" not in evidence_text
