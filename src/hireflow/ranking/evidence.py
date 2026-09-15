"""Build auditable, identity-free evidence from structured candidate profiles."""
from __future__ import annotations

from dataclasses import dataclass

from hireflow.models import CandidateProfile, EvidenceItem


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    section: str
    text: str


def build_evidence_catalog(profile: CandidateProfile) -> list[EvidenceRecord]:
    """Create a stable evidence catalog excluding name/contact/location metadata."""
    records: list[EvidenceRecord] = []

    def add(section: str, text: str) -> None:
        cleaned = " ".join(text.split()).strip()
        if cleaned:
            records.append(EvidenceRecord(f"E{len(records) + 1:03d}", section, cleaned))

    if profile.headline:
        add("headline", profile.headline)
    if profile.professional_summary:
        add("professional_summary", profile.professional_summary)
    for skill in profile.skills:
        add("technical_skills", skill)
    for software in profile.software:
        add("software", software)
    if profile.total_experience_years is not None:
        add("experience", f"{profile.total_experience_years:g}+ years of total experience")
    for exp in profile.work_experience:
        heading = exp.job_title + (f" at {exp.company}" if exp.company else "")
        add("work_experience", heading)
        for responsibility in exp.responsibilities:
            add("work_experience", responsibility)
        for achievement in exp.achievements:
            add("work_experience", achievement)
    for edu in profile.education:
        text = edu.degree
        if edu.field_of_study:
            text += f" in {edu.field_of_study}"
        add("education", text)
    for cert in profile.certifications:
        add("certifications", cert.name)
    return records


def record_to_evidence(record: EvidenceRecord, requirement: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=record.evidence_id,
        requirement=requirement,
        evidence=record.text,
        source_section=record.section,
    )
