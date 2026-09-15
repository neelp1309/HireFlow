#!/usr/bin/env python
"""Offline smoke validation for the Phase 5 product service."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.models import CandidateProfile, Education, JobRequirements, WorkExperience  # noqa: E402
from hireflow.product import HireFlowProductService, ProductSearchConfig  # noqa: E402


def candidate(candidate_id: str, role: str, years: float) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.pdf"),
        full_name=f"Candidate {candidate_id}",
        headline=role,
        professional_summary=f"Accounting professional with {years:g}+ years of experience.",
        skills=["Financial Reporting", "Account Reconciliation"],
        software=["Microsoft Excel", "QuickBooks"],
        total_experience_years=years,
        work_experience=[WorkExperience(job_title=role, responsibilities=["Prepared financial statements and reconciliations"])],
        education=[Education(degree="Bachelor of Science", field_of_study="Accounting")],
    )


job = JobRequirements(
    job_id="phase5_validation",
    title="Senior Accountant",
    experience_min_years=3,
    experience_max_years=5,
    responsibilities=["Prepare financial statements", "Prepare account reconciliations"],
    required_skills=["Financial reporting", "Account reconciliation"],
    required_software=["Microsoft Excel", "Accounting software"],
    education_requirements=["Bachelor's degree in Accounting or Finance"],
)

bundle = HireFlowProductService().search_structured(
    job,
    [candidate("senior", "Senior Accountant", 4), candidate("assistant", "Accounting Assistant", 1)],
    ProductSearchConfig(top_k_retrieval=2, top_k_rerank=2, final_top_k=2, vector_backend="numpy"),
)
assert len(bundle.results) == 2
assert bundle.results[0].profile.candidate_id == "senior"
assert bundle.results[0].match.final_score > bundle.results[1].match.final_score
print("HireFlow Phase 5 validation passed.")
