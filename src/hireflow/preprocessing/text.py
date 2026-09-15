"""Deterministic text cleaning used before parsing and embedding."""
from __future__ import annotations

import re

BULLET = "•"
PLACEHOLDER_PATTERNS = (
    re.compile(r"\{generate_achievements\([^}]*\)\}", re.IGNORECASE),
)


def normalize_pdf_text(text: str) -> str:
    """Normalize extracted PDF text without inventing content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # The supplied PDFs use DEL (0x7f) as their extracted bullet character.
    text = text.replace("\x7f", f"\n{BULLET} ")
    text = text.replace("●", BULLET).replace("▪", BULLET).replace("◦", BULLET)
    for pattern in PLACEHOLDER_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_bullets(text: str) -> list[str]:
    """Return bullet items, joining wrapped lines within each bullet."""
    if BULLET not in text:
        return []
    items = []
    for chunk in text.split(BULLET)[1:]:
        value = collapse_whitespace(chunk)
        if value:
            items.append(value)
    return items
