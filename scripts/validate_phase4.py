#!/usr/bin/env python
"""Smoke validation for Phase 4 benchmark and experiment outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.evaluation import benchmark_summary, load_benchmark  # noqa: E402


def main() -> None:
    benchmark_path = PROJECT_ROOT / "data" / "evaluation" / "senior_accountant_benchmark.csv"
    results_path = PROJECT_ROOT / "artifacts" / "evaluation" / "phase4_experiment_results.json"
    records = load_benchmark(benchmark_path, expected_job_id="senior_accountant_001")
    summary = benchmark_summary(records)
    assert summary["candidate_count"] == 50
    assert summary["label_distribution"] == {
        "not_suitable": 7,
        "weak_fit": 10,
        "moderate_fit": 20,
        "strong_fit": 13,
    }
    assert results_path.exists(), "Run scripts/evaluate_phase4.py before Phase 4 validation"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(payload["experiments"]) == 4
    for experiment in payload["experiments"]:
        assert len(experiment["ranked_candidate_ids"]) == 50
        metrics = experiment["metrics"]
        assert "ndcg_at_k" in metrics and "10" in metrics["ndcg_at_k"]
        assert 0 <= metrics["ndcg_at_k"]["10"] <= 1
    print("HireFlow Phase 4 validation passed.")


if __name__ == "__main__":
    main()
