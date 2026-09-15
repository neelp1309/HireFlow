from pathlib import Path

from hireflow.models import CandidateProfile, Education, JobRequirements, WorkExperience
from hireflow.product import HireFlowProductService, ProductSearchConfig, results_to_csv, results_to_json


def _candidate(candidate_id: str, role: str, years: float, skills: list[str], software: list[str]) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.pdf"),
        full_name=f"Candidate {candidate_id}",
        headline=role,
        professional_summary=f"Accounting professional with {years:g}+ years of experience in financial reporting and reconciliation.",
        skills=skills,
        software=software,
        total_experience_years=years,
        work_experience=[
            WorkExperience(
                job_title=role,
                company="Example Co",
                responsibilities=["Prepared financial statements", "Performed account reconciliations"],
            )
        ],
        education=[Education(degree="Bachelor of Science", field_of_study="Accounting")],
    )


def test_phase5_service_returns_ranked_exportable_product_bundle() -> None:
    job = JobRequirements(
        job_id="j1",
        title="Senior Accountant",
        experience_min_years=3,
        experience_max_years=5,
        responsibilities=["Prepare financial statements", "Prepare account reconciliations"],
        required_skills=["Financial reporting", "Account reconciliation"],
        required_software=["Microsoft Excel", "Accounting software"],
        education_requirements=["Bachelor's degree in Accounting or Finance"],
    )
    candidates = [
        _candidate("strong", "Senior Accountant", 4, ["Financial Reporting", "Account Reconciliation"], ["Microsoft Excel", "QuickBooks"]),
        _candidate("junior", "Accounting Assistant", 1, ["Accounts Payable"], ["Microsoft Excel"]),
        _candidate("analyst", "Financial Analyst", 3, ["Financial Analysis"], ["Microsoft Excel"]),
    ]
    service = HireFlowProductService()
    bundle = service.search_structured(
        job,
        candidates,
        ProductSearchConfig(
            top_k_retrieval=3,
            top_k_rerank=3,
            final_top_k=2,
            vector_backend="numpy",
        ),
    )

    assert bundle.candidate_count == 3
    assert len(bundle.results) == 2
    assert bundle.results[0].profile.candidate_id == "strong"
    assert bundle.results[0].match.final_score >= bundle.results[1].match.final_score
    assert "candidate_name" in results_to_csv(bundle)
    json_output = results_to_json(bundle)
    assert "Candidate strong" in json_output
    assert "contact" not in json_output
