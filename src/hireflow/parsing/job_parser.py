"""Deterministic parser for the supplied Senior Accountant job description."""
from __future__ import annotations

import re
from pathlib import Path

from hireflow.ingestion import extract_pdf_text
from hireflow.models.schemas import JobRequirements
from hireflow.preprocessing import JOB_HEADINGS, extract_bullets, normalize_pdf_text, split_sections

_EXP_RE = re.compile(r"Experience Required:\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*years", re.IGNORECASE)


def _metadata(preamble: str) -> dict[str, str | float | None]:
    lines = [line.strip() for line in preamble.splitlines() if line.strip()]
    title = (lines[0] if lines else "Untitled Position").replace(" POSITION", "").title()
    company = re.search(r"Company:\s*(.*?)\s+Location:", preamble, re.IGNORECASE | re.DOTALL)
    location = re.search(r"Location:\s*(.*?)\s+Employment\s+Type:", preamble, re.IGNORECASE | re.DOTALL)
    employment = re.search(r"Employment\s+Type:\s*(.*?)\s+Salary Range:", preamble, re.IGNORECASE | re.DOTALL)
    exp = _EXP_RE.search(preamble)
    return {
        "title": title,
        "company": company.group(1).strip() if company else None,
        "location": location.group(1).strip() if location else None,
        "employment_type": employment.group(1).strip() if employment else None,
        "experience_min_years": float(exp.group(1)) if exp else None,
        "experience_max_years": float(exp.group(2)) if exp else None,
    }


def _classify_required(items: list[str]) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    required_skills: list[str] = []
    required_software: list[str] = []
    education: list[str] = []
    preferred_certifications: list[str] = []
    preferred_qualifications: list[str] = []

    for item in items:
        lower = item.casefold()
        if "bachelor" in lower or "degree" in lower:
            education.append(item)
        elif "years" in lower and "experience" in lower:
            # Already represented as numeric experience metadata.
            continue
        elif "cpa" in lower and "preferred" in lower:
            preferred_certifications.append("CPA")
        elif "microsoft excel" in lower:
            required_software.append("Microsoft Excel")
        elif "accounting software" in lower:
            required_software.append("Accounting software (QuickBooks, SAP, Oracle, or similar)")
        else:
            required_skills.append(item)
    return required_skills, required_software, education, preferred_certifications, preferred_qualifications


def parse_job_pdf(path: str | Path, job_id: str = "senior_accountant_001") -> JobRequirements:
    parsed = extract_pdf_text(path)
    cleaned = normalize_pdf_text(parsed.text)
    sections = split_sections(cleaned, JOB_HEADINGS)
    meta = _metadata(sections.get("PREAMBLE", ""))

    responsibilities = extract_bullets(sections.get("RESPONSIBILITIES", ""))
    required_items = extract_bullets(sections.get("REQUIRED SKILLS", ""))
    required_skills, required_software, education, preferred_certs, preferred_from_required = _classify_required(required_items)
    preferred = extract_bullets(sections.get("PREFERRED QUALIFICATIONS", ""))
    technical = extract_bullets(sections.get("TECHNICAL SKILLS", ""))
    soft = extract_bullets(sections.get("SOFT SKILLS", ""))

    return JobRequirements(
        job_id=job_id,
        source_file=parsed.source_file,
        title=str(meta["title"]),
        company=meta["company"],
        location=meta["location"],
        employment_type=meta["employment_type"],
        experience_min_years=meta["experience_min_years"],
        experience_max_years=meta["experience_max_years"],
        responsibilities=responsibilities,
        required_skills=required_skills,
        required_software=required_software,
        preferred_skills=technical,
        preferred_qualifications=preferred_from_required + preferred,
        education_requirements=education,
        preferred_certifications=preferred_certs,
        soft_skills=soft,
        raw_text=cleaned,
    )
