"""Smoke validation for Phase 6 production engineering components."""
from __future__ import annotations

import tempfile
from pathlib import Path

from hireflow.api.app import create_app
from hireflow.config import Settings
from hireflow.models import CandidateProfile, Education, JobRequirements, WorkExperience
from hireflow.product import HireFlowProductService, ProductSearchConfig
from hireflow.production import CorpusIndexManager


def candidate(candidate_id: str, role: str, years: float) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.pdf"),
        headline=role,
        professional_summary=f"Accounting professional with {years:g} years experience",
        skills=["Financial Reporting", "Account Reconciliation"],
        software=["Microsoft Excel", "QuickBooks"],
        total_experience_years=years,
        work_experience=[
            WorkExperience(
                job_title=role,
                company="Example Co",
                responsibilities=["Prepared financial statements", "Performed reconciliations"],
            )
        ],
        education=[Education(degree="Bachelor of Science", field_of_study="Accounting")],
    )


def main() -> None:
    job = JobRequirements(
        job_id="phase6",
        title="Senior Accountant",
        experience_min_years=3,
        experience_max_years=5,
        required_skills=["Financial reporting", "Account reconciliation"],
        required_software=["Microsoft Excel", "Accounting software"],
        education_requirements=["Bachelor's degree in Accounting or Finance"],
    )
    candidates = [candidate("c1", "Senior Accountant", 4), candidate("c2", "Accounting Assistant", 1)]
    with tempfile.TemporaryDirectory(prefix="hireflow_phase6_") as tmp:
        base = Path(tmp)
        settings = Settings(
            _env_file=None,
            app_env="test",
            artifacts_dir=base / "artifacts",
            vector_index_dir=base / "indexes",
            cache_dir=base / "cache",
            log_dir=base / "logs",
            search_cache_enabled=True,
        )
        service = HireFlowProductService(settings=settings)
        config = ProductSearchConfig(
            top_k_retrieval=2,
            top_k_rerank=2,
            final_top_k=1,
            vector_backend="numpy",
        )
        first = service.search_structured(job, candidates, config)
        second = service.search_structured(job, candidates, config)
        assert not first.cache_hit and second.cache_hit
        assert first.results[0].profile.candidate_id == "c1"
        assert "grounded_evaluation" in first.stage_timings_ms

        manager = CorpusIndexManager(settings)
        initial = manager.sync("phase6", candidates, vector_backend="numpy")
        reused = manager.sync("phase6", candidates, vector_backend="numpy")
        assert initial.report.embedded_documents == 2
        assert reused.report.embedded_documents == 0
        assert reused.report.reused_embeddings == 2

        app = create_app(settings)
        route_paths = {route.path for route in app.routes}
        required = {"/health/live", "/health/ready", "/metrics", "/v1/search/structured", "/v1/search/files"}
        assert required.issubset(route_paths)

    print("HireFlow Phase 6 validation passed.")


if __name__ == "__main__":
    main()
