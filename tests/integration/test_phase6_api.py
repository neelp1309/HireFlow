from pathlib import Path

from fastapi.testclient import TestClient

from hireflow.api.app import create_app
from hireflow.config import Settings
from hireflow.models import CandidateProfile, Education, JobRequirements, WorkExperience
from hireflow.product import ProductSearchConfig


def candidate(candidate_id: str, role: str, years: float) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.pdf"),
        full_name=f"Private Name {candidate_id}",
        headline=role,
        professional_summary="Accounting professional with financial reporting experience",
        skills=["Financial Reporting", "Account Reconciliation"],
        software=["Microsoft Excel", "QuickBooks"],
        total_experience_years=years,
        work_experience=[
            WorkExperience(job_title=role, responsibilities=["Prepared financial statements"])
        ],
        education=[Education(degree="Bachelor of Science", field_of_study="Accounting")],
        raw_text="PRIVATE RESUME RAW TEXT",
    )


def test_phase6_api_health_search_metrics_and_privacy(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        artifacts_dir=tmp_path / "artifacts",
        vector_index_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        search_cache_enabled=True,
    )
    app = create_app(settings)
    job = JobRequirements(
        job_id="j1",
        title="Senior Accountant",
        experience_min_years=3,
        experience_max_years=5,
        required_skills=["Financial reporting", "Account reconciliation"],
        required_software=["Microsoft Excel", "Accounting software"],
        education_requirements=["Bachelor's degree in Accounting or Finance"],
    )
    payload = {
        "job": job.model_dump(mode="json"),
        "candidates": [
            candidate("strong", "Senior Accountant", 4).model_dump(mode="json"),
            candidate("junior", "Accounting Assistant", 1).model_dump(mode="json"),
        ],
        "config": ProductSearchConfig(
            top_k_retrieval=2,
            top_k_rerank=2,
            final_top_k=1,
            vector_backend="numpy",
        ).model_dump(mode="json"),
    }
    with TestClient(app) as client:
        live = client.get("/health/live")
        assert live.status_code == 200
        assert live.headers["x-request-id"]
        assert client.get("/health/ready").status_code == 200

        response = client.post("/v1/search/structured", json=payload, headers={"X-Request-ID": "test-request"})
        assert response.status_code == 200
        body = response.json()
        assert body["request_id"] == "test-request"
        assert body["results"][0]["candidate_id"] == "strong"
        serialized = response.text
        assert "Private Name" not in serialized
        assert "PRIVATE RESUME RAW TEXT" not in serialized
        assert body["cache_hit"] is False

        cached = client.post("/v1/search/structured", json=payload)
        assert cached.status_code == 200
        assert cached.json()["cache_hit"] is True

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "hireflow_http_requests_total" in metrics.text
        assert "hireflow_search_requests_total" in metrics.text


def test_api_key_can_protect_search_routes(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        api_access_key="secret",
        artifacts_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        response = client.post("/v1/search/structured", json={})
        assert response.status_code == 401
