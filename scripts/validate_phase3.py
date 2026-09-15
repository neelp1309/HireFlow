#!/usr/bin/env python
"""Smoke validation for Phase 3 intelligence components on the supplied dataset."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.evaluation.gemini import _build_prompt  # noqa: E402
from hireflow.models import CandidateProfile, JobRequirements  # noqa: E402
from hireflow.utils import read_json, read_jsonl  # noqa: E402


def main() -> None:
    required_private = [
        PROJECT_ROOT / "data" / "processed" / "candidates.jsonl",
        PROJECT_ROOT / "data" / "processed" / "job.json",
        PROJECT_ROOT / "artifacts" / "indexes" / "baseline_full_tfidf" / "manifest.json",
    ]
    if not all(path.exists() for path in required_private):
        print(
            "HireFlow Phase 3 validation skipped: private processed dataset/index artifacts "
            "are intentionally absent from the sanitized repository."
        )
        return
    out = PROJECT_ROOT / "artifacts" / "phase3_validation_results.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "rank_candidates.py"),
        "--top-k-retrieval", "20",
        "--top-k-rerank", "10",
        "--final-top-k", "5",
        "--evaluator", "heuristic",
        "--output", str(out),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(out.read_text(encoding="utf-8"))
    results = payload["results"]
    assert len(results) == 10
    assert all(results[i]["final_score"] >= results[i + 1]["final_score"] for i in range(len(results) - 1))

    top5_roles = [str(item["headline"]).casefold() for item in results[:5]]
    assert sum("accountant" in role for role in top5_roles) >= 3, top5_roles
    assert "tax manager" not in top5_roles[:2], top5_roles

    candidates = read_jsonl(PROJECT_ROOT / "data" / "processed" / "candidates.jsonl", CandidateProfile)
    candidate_map = {item.candidate_id: item for item in candidates}
    job = read_json(PROJECT_ROOT / "data" / "processed" / "job.json", JobRequirements)

    # Verify LLM prompt construction never includes candidate identity/contact fields.
    first = results[0]
    match_payload = first
    from hireflow.models import CandidateMatch
    match = CandidateMatch(
        candidate_id=first["candidate_id"],
        rank=first["rerank_rank"],
        retrieval_score=first["retrieval_score"],
        component_scores=first["component_scores"],
        final_score=first["final_score"],
        matched_requirements=first["matched_requirements"],
        missing_requirements=first["missing_requirements"],
    )
    profile = candidate_map[first["candidate_id"]]
    prompt, _ = _build_prompt(profile, job, match)
    assert not profile.full_name or profile.full_name not in prompt
    assert not profile.contact or not profile.contact.email or profile.contact.email not in prompt
    assert not profile.contact or not profile.contact.phone or profile.contact.phone not in prompt

    print("HireFlow Phase 3 validation passed.")
    print("Top 5 reranked candidates:")
    for item in results[:5]:
        print(
            f"  #{item['rerank_rank']} {item['candidate_id']} | {item['headline']} | "
            f"{item['experience_years']} years | score={item['final_score']} | "
            f"retrieval_rank={item['retrieval_rank']}"
        )


if __name__ == "__main__":
    main()
