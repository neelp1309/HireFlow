from pathlib import Path

from hireflow.evaluation import HeuristicGroundedEvaluator
from hireflow.models import CandidateProfile, CandidateRetrievalResult, JobRequirements, RetrievalHit
from hireflow.ranking import HybridCandidateReranker, MatchingTaxonomy, RequirementMatcher


def _result(candidate_id: str, score: float) -> CandidateRetrievalResult:
    return CandidateRetrievalResult(
        candidate_id=candidate_id,
        retrieval_score=score,
        best_section="full_resume",
        hits=[
            RetrievalHit(
                vector_id=f"{candidate_id}:full",
                candidate_id=candidate_id,
                section="full_resume",
                score=score,
                text="candidate text",
            )
        ],
    )


def test_requirement_aware_reranking_corrects_semantic_only_ordering() -> None:
    job = JobRequirements(
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
        required_software=["Microsoft Excel", "Accounting software (QuickBooks, SAP, Oracle, or similar)"],
        education_requirements=["Bachelor's degree in Accounting or Finance"],
    )
    good = CandidateProfile(
        candidate_id="good",
        source_file=Path("good.pdf"),
        headline="Senior Accountant",
        professional_summary="Financial reporting accountant.",
        skills=["Financial Reporting", "Month-End Closing", "Journal Entries", "Bank Reconciliation", "Tax Preparation"],
        software=["Microsoft Excel", "QuickBooks"],
        total_experience_years=4,
        education=[{"degree": "Bachelor of Science", "field_of_study": "Accounting"}],
    )
    adjacent = CandidateProfile(
        candidate_id="adjacent",
        source_file=Path("adjacent.pdf"),
        headline="Tax Manager",
        professional_summary="Tax specialist.",
        skills=["Tax Preparation", "Tax Compliance"],
        software=["Microsoft Excel"],
        total_experience_years=5,
        education=[{"degree": "Bachelor of Science", "field_of_study": "Accounting"}],
    )

    # Semantic retrieval alone has the adjacent role first.
    retrieved = [_result("adjacent", 0.50), _result("good", 0.35)]
    candidates = {"good": good, "adjacent": adjacent}
    taxonomy_path = Path(__file__).resolve().parents[2] / "configs" / "matching_taxonomy.yaml"
    reranker = HybridCandidateReranker(RequirementMatcher(MatchingTaxonomy.load(taxonomy_path)))
    ranked = reranker.rerank(retrieved, candidates, job)

    assert ranked[0].candidate_id == "good"
    evaluation = HeuristicGroundedEvaluator().evaluate(good, job, ranked[0])
    assert evaluation.overall_score == ranked[0].final_score
    assert evaluation.evidence
