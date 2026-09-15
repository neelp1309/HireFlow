"""Create retrieval documents from structured candidate profiles."""
from __future__ import annotations

from hireflow.models import CandidateProfile, RetrievalDocument


def candidate_documents(profile: CandidateProfile, mode: str = "full") -> list[RetrievalDocument]:
    """Convert a candidate into one or more retrievable text units.

    ``full`` is the baseline: one qualification-only representation per resume.
    ``section`` is available for the next retrieval experiment and groups related
    resume sections without splitting arbitrary character windows.
    """
    if mode not in {"full", "section"}:
        raise ValueError("mode must be 'full' or 'section'")

    common_metadata = {
        "headline": profile.headline,
        "experience_years": profile.total_experience_years,
        "source_file": profile.source_file.name,
    }

    if mode == "full":
        return [
            RetrievalDocument(
                vector_id=f"{profile.candidate_id}:full",
                candidate_id=profile.candidate_id,
                section="full_resume",
                text=profile.matching_text(),
                metadata=common_metadata,
            )
        ]

    docs: list[RetrievalDocument] = []
    if profile.headline or profile.professional_summary:
        text = "\n".join(
            part
            for part in (
                f"Role: {profile.headline}" if profile.headline else "",
                f"Professional Summary: {profile.professional_summary}" if profile.professional_summary else "",
            )
            if part
        )
        docs.append(RetrievalDocument(vector_id=f"{profile.candidate_id}:summary", candidate_id=profile.candidate_id, section="summary", text=text, metadata=common_metadata))

    if profile.skills or profile.software:
        pieces = []
        if profile.skills:
            pieces.append("Skills: " + ", ".join(profile.skills))
        if profile.software:
            pieces.append("Software: " + ", ".join(profile.software))
        docs.append(RetrievalDocument(vector_id=f"{profile.candidate_id}:skills", candidate_id=profile.candidate_id, section="skills_software", text="\n".join(pieces), metadata=common_metadata))

    if profile.work_experience:
        pieces = []
        for exp in profile.work_experience:
            line = f"{exp.job_title}"
            if exp.company:
                line += f" at {exp.company}"
            if exp.responsibilities:
                line += ". " + " ".join(exp.responsibilities)
            pieces.append(line)
        docs.append(RetrievalDocument(vector_id=f"{profile.candidate_id}:experience", candidate_id=profile.candidate_id, section="experience", text="\n".join(pieces), metadata=common_metadata))

    if profile.education or profile.certifications:
        pieces = []
        for edu in profile.education:
            line = edu.degree + (f" in {edu.field_of_study}" if edu.field_of_study else "")
            pieces.append("Education: " + line)
        if profile.certifications:
            pieces.append("Certifications: " + ", ".join(c.name for c in profile.certifications))
        docs.append(RetrievalDocument(vector_id=f"{profile.candidate_id}:education", candidate_id=profile.candidate_id, section="education_certifications", text="\n".join(pieces), metadata=common_metadata))

    return docs
