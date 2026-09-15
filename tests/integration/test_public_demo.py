import json
from pathlib import Path

from hireflow.api.schemas import StructuredSearchRequest
from hireflow.product import HireFlowProductService


ROOT = Path(__file__).resolve().parents[2]


def test_public_demo_ranks_strong_candidate_first() -> None:
    payload = json.loads((ROOT / "examples/structured_search_request.json").read_text(encoding="utf-8"))
    request = StructuredSearchRequest.model_validate(payload)
    bundle = HireFlowProductService().search_structured(request.job, request.candidates, request.config)
    assert bundle.results[0].profile.candidate_id == "demo_candidate_01"
    assert bundle.results[0].evaluation.overall_score > bundle.results[-1].evaluation.overall_score
