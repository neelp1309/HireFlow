from __future__ import annotations

from pathlib import Path

import pytest

from hireflow.evaluation import benchmark_labels, load_benchmark
from hireflow.evaluation.ranking_metrics import evaluate_ranking


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_benchmark_has_all_50_candidates() -> None:
    benchmark = PROJECT_ROOT / "data" / "evaluation" / "senior_accountant_benchmark.csv"
    records = load_benchmark(benchmark, expected_job_id="senior_accountant_001")
    labels = benchmark_labels(records)
    assert len(labels) == 50
    assert sum(label == 3 for label in labels.values()) == 13


def test_ideal_project_ranking_scores_one_ndcg() -> None:
    benchmark = PROJECT_ROOT / "data" / "evaluation" / "senior_accountant_benchmark.csv"
    records = load_benchmark(benchmark, expected_job_id="senior_accountant_001")
    labels = benchmark_labels(records)
    ideal = sorted(labels, key=lambda cid: (-labels[cid], cid))
    metrics = evaluate_ranking(ideal, labels, ks=(5, 10, 20))
    assert metrics.ndcg_at_k[5] == pytest.approx(1.0)
    assert metrics.ndcg_at_k[10] == pytest.approx(1.0)
