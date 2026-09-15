"""Final release-hygiene validation for the public HireFlow repository."""
from __future__ import annotations

import json
import re
from pathlib import Path

from hireflow import __version__
from hireflow.api.schemas import StructuredSearchRequest
from hireflow.product import HireFlowProductService

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "SECURITY.md",
    "docs/FINAL_RELEASE_CHECKLIST.md",
    "examples/structured_search_request.json",
    "docs/RESULTS.md",
    "docs/INTERVIEW_GUIDE.md",
    "docs/RESPONSIBLE_AI.md",
    "docs/DEPLOYMENT.md",
]

FORBIDDEN_RELEASE_PATHS = [
    ROOT / ".env",
]

SECRET_ASSIGNMENT = re.compile(r"^\s*GEMINI_API_KEY\s*=\s*([^\s#]+)", re.MULTILINE)
ALLOWED_SECRET_PLACEHOLDERS = {
    "your_key_here",
    "replace_with_your_key",
    "<secret>",
    "<your_key>",
    "<your-key>",
}


def main() -> None:
    assert __version__ == "1.0.0", f"Expected version 1.0.0, got {__version__}"

    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    assert not missing, f"Missing release files: {missing}"

    for path in FORBIDDEN_RELEASE_PATHS:
        assert not path.exists(), f"Forbidden release file exists: {path}"

    for private_dir in [ROOT / "data/raw/jobs", ROOT / "data/raw/resumes", ROOT / "data/processed"]:
        leaked = [p for p in private_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
        assert not leaked, f"Private data leaked into release: {leaked[:3]}"

    index_dir = ROOT / "artifacts/indexes"
    leaked_indexes = [p for p in index_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    assert not leaked_indexes, f"Runtime indexes leaked into release: {leaked_indexes[:3]}"

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".zip"}:
            continue
        if any(part in {".git", ".venv"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in SECRET_ASSIGNMENT.finditer(text):
            value = match.group(1).strip()
            assert value in ALLOWED_SECRET_PLACEHOLDERS, (
                f"Potential secret detected in {path.relative_to(ROOT)}"
            )

    payload = json.loads((ROOT / "examples/structured_search_request.json").read_text(encoding="utf-8"))
    request = StructuredSearchRequest.model_validate(payload)
    bundle = HireFlowProductService().search_structured(request.job, request.candidates, request.config)
    assert bundle.results
    assert bundle.results[0].profile.candidate_id == "demo_candidate_01"

    print("HireFlow v1.0.0 release validation passed.")


if __name__ == "__main__":
    main()
