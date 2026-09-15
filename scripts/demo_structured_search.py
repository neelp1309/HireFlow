"""Run a public, PII-free HireFlow demo without the original resume dataset."""
from __future__ import annotations

import json
from pathlib import Path

from hireflow.api.schemas import StructuredSearchRequest
from hireflow.product import HireFlowProductService


ROOT = Path(__file__).resolve().parents[1]
DEMO_REQUEST = ROOT / "examples" / "structured_search_request.json"


def main() -> None:
    payload = json.loads(DEMO_REQUEST.read_text(encoding="utf-8"))
    request = StructuredSearchRequest.model_validate(payload)
    service = HireFlowProductService()
    bundle = service.search_structured(request.job, request.candidates, request.config)

    print("HireFlow public structured-search demo")
    print(f"Job: {bundle.job.title}")
    print(f"Candidates: {bundle.candidate_count}")
    print(f"Embedding: {bundle.embedding_model}")
    print(f"Vector backend: {bundle.vector_backend}")
    print()
    print(f"{'Rank':<6}{'Candidate':<24}{'Role':<24}{'Score':>8}{'Recommendation':>22}")
    print("-" * 84)
    for item in bundle.results:
        print(
            f"{item.match.rank:<6}"
            f"{item.profile.candidate_id:<24}"
            f"{(item.profile.headline or 'Unknown'):<24}"
            f"{item.evaluation.overall_score:>8.2f}"
            f"{item.evaluation.recommendation.value:>22}"
        )

    if not bundle.results or bundle.results[0].profile.candidate_id != "demo_candidate_01":
        raise SystemExit("Demo validation failed: expected demo_candidate_01 to rank first")
    print("\nDemo validation passed: the strongest accounting candidate ranked first.")


if __name__ == "__main__":
    main()
