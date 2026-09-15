"""PDF text ingestion with clear, typed failures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from hireflow.exceptions import DocumentParsingError


@dataclass(frozen=True, slots=True)
class ParsedPdf:
    source_file: Path
    text: str
    page_count: int


def extract_pdf_text(path: str | Path) -> ParsedPdf:
    """Extract text from a text-based PDF.

    OCR is intentionally out of scope for the baseline pipeline. A document
    with no extractable text raises a clear error so a later OCR fallback can
    be introduced without silently producing empty candidate profiles.
    """
    source = Path(path)
    if not source.exists():
        raise DocumentParsingError(f"PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise DocumentParsingError(f"Unsupported document type: {source.suffix}")

    try:
        reader = PdfReader(str(source), strict=False)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # pypdf exposes several parser-specific exceptions
        raise DocumentParsingError(f"Failed to read PDF {source.name}: {exc}") from exc

    text = "\n".join(page for page in pages if page).strip()
    if not text:
        raise DocumentParsingError(
            f"No extractable text found in {source.name}. The file may require OCR."
        )
    return ParsedPdf(source_file=source, text=text, page_count=len(reader.pages))
