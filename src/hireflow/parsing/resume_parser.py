"""Deterministic resume parser for the baseline HireFlow dataset."""
from __future__ import annotations

import re
from pathlib import Path

from hireflow.ingestion import extract_pdf_text
from hireflow.models.schemas import CandidateProfile, Certification, ContactInfo, Education, WorkExperience
from hireflow.preprocessing import RESUME_HEADINGS, collapse_whitespace, extract_bullets, normalize_pdf_text, prose, split_sections

_EXPERIENCE_RE = re.compile(r"(?P<years>\d+(?:\.\d+)?)\+\s*years", re.IGNORECASE)
_EMAIL_RE = re.compile(r"Email:\s*([^\n]+)", re.IGNORECASE)
_PHONE_RE = re.compile(r"Phone:\s*([^\n]+)", re.IGNORECASE)
_LOCATION_RE = re.compile(r"Location:\s*([^\n]+)", re.IGNORECASE)
_LINKEDIN_RE = re.compile(r"LinkedIn:\s*([^\n]+)", re.IGNORECASE)
_RESUME_NUMBER_RE = re.compile(r"Resume_(\d+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _candidate_id(path: Path) -> str:
    match = _RESUME_NUMBER_RE.search(path.stem)
    return f"resume_{int(match.group(1)):02d}" if match else path.stem.lower().replace(" ", "_")


def _parse_preamble(preamble: str) -> tuple[str | None, str | None, ContactInfo | None, str | None]:
    lines = [line.strip() for line in preamble.splitlines() if line.strip()]
    full_name = lines[0].title() if lines else None
    headline = lines[1] if len(lines) > 1 and not lines[1].lower().startswith("contact") else None

    email = (_EMAIL_RE.search(preamble).group(1).strip() if _EMAIL_RE.search(preamble) else None)
    phone = (_PHONE_RE.search(preamble).group(1).strip() if _PHONE_RE.search(preamble) else None)
    location = (_LOCATION_RE.search(preamble).group(1).strip() if _LOCATION_RE.search(preamble) else None)
    linkedin = (_LINKEDIN_RE.search(preamble).group(1).strip() if _LINKEDIN_RE.search(preamble) else None)
    contact = ContactInfo(email=email, phone=phone, linkedin=linkedin) if any((email, phone, linkedin)) else None
    return full_name, headline, contact, location


def _parse_work_experience(section: str) -> list[WorkExperience]:
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    experiences: list[WorkExperience] = []
    current_header: str | None = None
    current_bullets: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_bullets
        if not current_header:
            return
        parts = [collapse_whitespace(p) for p in current_header.split("|")]
        title = parts[0] if parts else "Unknown Role"
        company = parts[1] if len(parts) > 1 else None
        date_text = parts[2] if len(parts) > 2 else None
        start_date = end_date = None
        if date_text:
            if " - " in date_text:
                start_date, end_date = [p.strip() for p in date_text.split(" - ", 1)]
            else:
                start_date = date_text
        experiences.append(
            WorkExperience(
                job_title=title,
                company=company,
                start_date=start_date,
                end_date=end_date,
                responsibilities=current_bullets.copy(),
            )
        )
        current_header = None
        current_bullets = []

    for line in lines:
        if line.startswith("•"):
            item = collapse_whitespace(line.lstrip("• "))
            if item:
                current_bullets.append(item)
        elif line.count("|") >= 2:
            flush()
            current_header = line
        elif current_header and current_bullets:
            # Wrapped responsibility line.
            current_bullets[-1] = collapse_whitespace(current_bullets[-1] + " " + line)
        elif current_header:
            current_header += " " + line
    flush()
    return experiences


def _parse_education(section: str) -> list[Education]:
    lines = [line.strip() for line in section.splitlines() if line.strip() and not line.upper().startswith("GPA:")]
    if not lines:
        return []
    degree_line = lines[0]
    institution = None
    year = None
    if len(lines) > 1:
        parts = [p.strip() for p in lines[1].split("|")]
        institution = parts[0] or None
        match = _YEAR_RE.search(lines[1])
        year = int(match.group(1)) if match else None

    field = None
    degree = degree_line
    degree_match = re.match(r"(.+?)\s+in\s+(.+)", degree_line, flags=re.IGNORECASE)
    if degree_match:
        degree = degree_match.group(1).strip()
        field = degree_match.group(2).strip()
    return [Education(degree=degree, field_of_study=field, institution=institution, graduation_year=year)]


def _total_experience(summary: str | None) -> float | None:
    if not summary:
        return None
    match = _EXPERIENCE_RE.search(summary)
    return float(match.group("years")) if match else None


def parse_resume_pdf(path: str | Path) -> CandidateProfile:
    parsed = extract_pdf_text(path)
    cleaned = normalize_pdf_text(parsed.text)
    sections = split_sections(cleaned, RESUME_HEADINGS)

    full_name, headline, contact, location = _parse_preamble(sections.get("PREAMBLE", ""))
    summary = prose(sections.get("PROFESSIONAL SUMMARY", "")) or None
    skills = extract_bullets(sections.get("TECHNICAL SKILLS", ""))
    software = extract_bullets(sections.get("SOFTWARE PROFICIENCY", ""))
    experience = _parse_work_experience(sections.get("PROFESSIONAL EXPERIENCE", ""))
    education = _parse_education(sections.get("EDUCATION", ""))
    certifications = [Certification(name=item) for item in extract_bullets(sections.get("CERTIFICATIONS", ""))]
    achievements = extract_bullets(sections.get("ACHIEVEMENTS", ""))

    return CandidateProfile(
        candidate_id=_candidate_id(parsed.source_file),
        source_file=parsed.source_file,
        full_name=full_name,
        headline=headline,
        location=location,
        contact=contact,
        professional_summary=summary,
        skills=skills,
        software=software,
        work_experience=experience,
        total_experience_years=_total_experience(summary),
        education=education,
        certifications=certifications,
        achievements=achievements,
        raw_text=cleaned,
        parser_version="2.0-deterministic",
    )
