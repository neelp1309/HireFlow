"""Section-aware splitting for the supplied resume and JD formats."""
from __future__ import annotations

import re
from collections.abc import Iterable

from hireflow.preprocessing.text import collapse_whitespace


RESUME_HEADINGS = (
    "PROFESSIONAL SUMMARY",
    "TECHNICAL SKILLS",
    "SOFTWARE PROFICIENCY",
    "PROFESSIONAL EXPERIENCE",
    "EDUCATION",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
    "REFERENCES",
)

JOB_HEADINGS = (
    "JOB DESCRIPTION",
    "RESPONSIBILITIES",
    "REQUIRED SKILLS",
    "PREFERRED QUALIFICATIONS",
    "TECHNICAL SKILLS",
    "SOFT SKILLS",
    "BENEFITS",
    "ABOUT TECHFLOW SOLUTIONS",
    "APPLICATION PROCESS",
)


def split_sections(text: str, headings: Iterable[str]) -> dict[str, str]:
    """Split a normalized document by known headings.

    The preamble is returned under ``PREAMBLE``. Missing headings are simply
    absent, which keeps the parser robust to partially populated resumes.
    """
    heading_list = list(headings)
    escaped = "|".join(re.escape(h) for h in sorted(heading_list, key=len, reverse=True))
    pattern = re.compile(rf"(?m)^({escaped})\s*$")
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    if not matches:
        return {"PREAMBLE": text.strip()}

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["PREAMBLE"] = preamble
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[start:end].strip()
    return sections


def prose(text: str) -> str:
    """Collapse line wraps for prose-oriented sections."""
    return collapse_whitespace(text)
