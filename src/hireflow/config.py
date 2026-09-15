"""Centralized, environment-driven configuration for HireFlow."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``.

    Secrets are never committed to source control. Paths default to directories
    inside the repository so local development works out of the box.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "HireFlow"
    app_env: str = "development"
    log_level: str = "INFO"

    # API / production runtime
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1, le=16)
    api_access_key: str | None = Field(default=None, repr=False)
    api_max_request_mb: int = Field(default=25, ge=1, le=250)
    max_candidates_per_request: int = Field(default=500, ge=1, le=5000)
    metrics_enabled: bool = True

    # Process-local search cache. Cached bundles are never persisted to disk.
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: int = Field(default=900, ge=1, le=86400)
    search_cache_max_entries: int = Field(default=128, ge=1, le=5000)

    # Hosted-model resilience
    gemini_max_retries: int = Field(default=3, ge=1, le=6)
    gemini_initial_backoff_seconds: float = Field(default=1.0, ge=0.0, le=30.0)

    # Data and artifacts
    data_dir: Path = PROJECT_ROOT / "data"
    raw_resume_dir: Path = PROJECT_ROOT / "data" / "raw" / "resumes"
    raw_job_dir: Path = PROJECT_ROOT / "data" / "raw" / "jobs"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    vector_index_dir: Path = PROJECT_ROOT / "artifacts" / "indexes"
    cache_dir: Path = PROJECT_ROOT / "artifacts" / "cache"
    log_dir: Path = PROJECT_ROOT / "artifacts" / "logs"

    # Gemini configuration
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Retrieval defaults. These will be tuned through experiments later.
    top_k_retrieval: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=10, ge=1, le=50)
    final_top_k: int = Field(default=5, ge=1, le=25)
    min_retrieval_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Hybrid scoring weights. Kept in configuration so experiments do not
    # require source-code changes.
    weight_semantic: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_required_skills: float = Field(default=0.30, ge=0.0, le=1.0)
    weight_experience: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_role: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_preferred: float = Field(default=0.10, ge=0.0, le=1.0)

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        normalized = value.casefold().strip()
        allowed = {"development", "test", "staging", "production"}
        if normalized not in allowed:
            raise ValueError(f"APP_ENV must be one of {sorted(allowed)}")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper().strip()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return normalized

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> "Settings":
        if self.top_k_rerank > self.top_k_retrieval:
            raise ValueError("top_k_rerank cannot exceed top_k_retrieval")
        if self.final_top_k > self.top_k_rerank:
            raise ValueError("final_top_k cannot exceed top_k_rerank")
        return self

    @model_validator(mode="after")
    def validate_scoring_weights(self) -> "Settings":
        total = (
            self.weight_semantic
            + self.weight_required_skills
            + self.weight_experience
            + self.weight_role
            + self.weight_preferred
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Hybrid scoring weights must sum to 1.0; got {total:.6f}")
        return self

    def ensure_directories(self) -> None:
        """Create local runtime directories if they do not already exist."""
        for path in (
            self.raw_resume_dir,
            self.raw_job_dir,
            self.processed_dir,
            self.evaluation_dir,
            self.artifacts_dir,
            self.vector_index_dir,
            self.cache_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
