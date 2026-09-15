"""Smoke-test the Phase 1 project foundation."""

from pathlib import Path

from hireflow.config import get_settings
from hireflow.logging_config import configure_logging
from hireflow.models import CandidateProfile, JobRequirements


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)

    candidate = CandidateProfile(
        candidate_id="candidate_demo",
        source_file=Path("demo_resume.pdf"),
        headline="Senior Accountant",
        professional_summary="Accounting professional with financial reporting experience.",
        skills=["GAAP", "Financial Reporting", "GAAP"],
        software=["Excel", "QuickBooks"],
        total_experience_years=4,
    )
    job = JobRequirements(
        job_id="job_demo",
        title="Senior Accountant",
        experience_min_years=3,
        experience_max_years=5,
        required_skills=["GAAP", "Financial Reporting"],
        required_software=["Excel"],
    )

    assert candidate.skills == ["GAAP", "Financial Reporting"]
    assert "candidate_demo" not in candidate.matching_text()
    assert "Senior Accountant" in job.matching_text()

    print("HireFlow Phase 1 validation passed.")
    print(f"Project root: {settings.data_dir.parent}")
    print(f"Candidate matching text:\n{candidate.matching_text()}\n")
    print(f"Job matching text:\n{job.matching_text()}")


if __name__ == "__main__":
    main()
