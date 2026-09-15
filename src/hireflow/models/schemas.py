"""Validated domain schemas shared by ingestion, retrieval and evaluation."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model with strict handling of unexpected fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Recommendation(StrEnum):
    HIGHLY_RECOMMENDED = "highly_recommended"
    RECOMMENDED = "recommended"
    CONSIDER = "consider"
    NOT_RECOMMENDED = "not_recommended"


class ContactInfo(StrictModel):
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None


class WorkExperience(StrictModel):
    job_title: str
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_years: float | None = Field(default=None, ge=0)
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class Education(StrictModel):
    degree: str
    field_of_study: str | None = None
    institution: str | None = None
    graduation_year: int | None = Field(default=None, ge=1900, le=2100)


class Certification(StrictModel):
    name: str
    issuer: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)


class CandidateProfile(StrictModel):
    """Canonical structured representation of a resume.

    Personally identifying fields are kept as metadata for display, but the
    ``matching_text`` method intentionally excludes them from retrieval.
    """

    candidate_id: str
    source_file: Path
    full_name: str | None = None
    headline: str | None = None
    location: str | None = None
    contact: ContactInfo | None = None

    professional_summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    software: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    total_experience_years: float | None = Field(default=None, ge=0)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    raw_text: str = ""
    parser_version: str = "1.0"

    @field_validator("skills", "software", "achievements", mode="after")
    @classmethod
    def deduplicate_case_insensitive(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.casefold().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(value.strip())
        return result

    def matching_text(self) -> str:
        """Return qualification-related text while excluding candidate identity."""
        parts: list[str] = []
        if self.headline:
            parts.append(f"Role: {self.headline}")
        if self.professional_summary:
            parts.append(f"Professional Summary: {self.professional_summary}")
        if self.skills:
            parts.append("Skills: " + ", ".join(self.skills))
        if self.software:
            parts.append("Software: " + ", ".join(self.software))
        if self.total_experience_years is not None:
            parts.append(f"Total Experience: {self.total_experience_years:g} years")
        for item in self.work_experience:
            detail = f"Experience: {item.job_title}"
            if item.company:
                detail += f" at {item.company}"
            if item.responsibilities:
                detail += ". " + " ".join(item.responsibilities)
            if item.achievements:
                detail += ". " + " ".join(item.achievements)
            parts.append(detail)
        for item in self.education:
            text = f"Education: {item.degree}"
            if item.field_of_study:
                text += f" in {item.field_of_study}"
            parts.append(text)
        if self.certifications:
            parts.append("Certifications: " + ", ".join(c.name for c in self.certifications))
        return "\n".join(parts).strip()


class JobRequirements(StrictModel):
    job_id: str
    source_file: Path | None = None
    title: str
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None

    experience_min_years: float | None = Field(default=None, ge=0)
    experience_max_years: float | None = Field(default=None, ge=0)
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    required_software: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    preferred_certifications: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    raw_text: str = ""

    @field_validator(
        "responsibilities",
        "required_skills",
        "required_software",
        "preferred_skills",
        "preferred_qualifications",
        "education_requirements",
        "preferred_certifications",
        "soft_skills",
        mode="after",
    )
    @classmethod
    def deduplicate_case_insensitive(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.casefold().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(value.strip())
        return result

    def matching_text(self) -> str:
        """Return only job-relevant criteria, excluding benefits/contact boilerplate."""
        parts = [f"Job Title: {self.title}"]
        if self.experience_min_years is not None:
            exp = f"Experience: minimum {self.experience_min_years:g} years"
            if self.experience_max_years is not None:
                exp += f", preferred maximum {self.experience_max_years:g} years"
            parts.append(exp)
        if self.responsibilities:
            parts.append("Responsibilities: " + "; ".join(self.responsibilities))
        if self.required_skills:
            parts.append("Required Skills: " + ", ".join(self.required_skills))
        if self.required_software:
            parts.append("Required Software: " + ", ".join(self.required_software))
        if self.education_requirements:
            parts.append("Education: " + ", ".join(self.education_requirements))
        if self.preferred_skills:
            parts.append("Preferred Skills: " + ", ".join(self.preferred_skills))
        if self.preferred_qualifications:
            parts.append("Preferred Qualifications: " + ", ".join(self.preferred_qualifications))
        if self.preferred_certifications:
            parts.append("Preferred Certifications: " + ", ".join(self.preferred_certifications))
        return "\n".join(parts).strip()


class EvidenceItem(StrictModel):
    evidence_id: str | None = None
    requirement: str
    evidence: str
    source_section: str


class MatchComponentScores(StrictModel):
    semantic: float = Field(ge=0, le=100)
    required_skills: float = Field(ge=0, le=100)
    experience: float = Field(ge=0, le=100)
    role: float = Field(ge=0, le=100)
    preferred: float = Field(ge=0, le=100)


class CandidateMatch(StrictModel):
    candidate_id: str
    rank: int | None = Field(default=None, ge=1)
    retrieval_score: float = Field(ge=-1, le=1)
    component_scores: MatchComponentScores | None = None
    final_score: float | None = Field(default=None, ge=0, le=100)
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class CandidateEvaluation(StrictModel):
    candidate_id: str
    overall_score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class RetrievalDocument(StrictModel):
    """Text unit stored in the vector index."""

    vector_id: str
    candidate_id: str
    section: str
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RetrievalHit(StrictModel):
    vector_id: str
    candidate_id: str
    section: str
    score: float = Field(ge=-1.0, le=1.0)
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class CandidateRetrievalResult(StrictModel):
    candidate_id: str
    retrieval_score: float = Field(ge=-1.0, le=1.0)
    best_section: str
    hits: list[RetrievalHit] = Field(default_factory=list)
