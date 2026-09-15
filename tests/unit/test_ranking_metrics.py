from __future__ import annotations

import math

import pytest

from hireflow.evaluation.ranking_metrics import (
    average_precision,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    strong_fit_rate_at_k,
)


def test_binary_and_graded_metrics() -> None:
    labels = {"a": 3, "b": 0, "c": 2, "d": 1}
    ranking = ["a", "b", "c", "d"]

    assert precision_at_k(ranking, labels, 2) == 0.5
    assert recall_at_k(ranking, labels, 2) == 0.5
    assert strong_fit_rate_at_k(ranking, labels, 2) == 0.5
    assert reciprocal_rank(ranking, labels) == 1.0
    assert average_precision(ranking, labels) == pytest.approx((1.0 + 2 / 3) / 2)
    assert 0 < ndcg_at_k(ranking, labels, 4) <= 1


def test_perfect_ranking_has_ndcg_one() -> None:
    labels = {"a": 3, "b": 2, "c": 1, "d": 0}
    assert ndcg_at_k(["a", "b", "c", "d"], labels, 4) == pytest.approx(1.0)


def test_evaluate_ranking_returns_requested_ks() -> None:
    labels = {"a": 3, "b": 2, "c": 0}
    metrics = evaluate_ranking(["a", "b", "c"], labels, ks=(1, 2, 3))
    assert set(metrics.ndcg_at_k) == {1, 2, 3}
    assert metrics.relevant_count == 2
    assert metrics.strong_fit_count == 1
    assert metrics.candidate_count == 3


def test_duplicate_ranking_is_rejected() -> None:
    labels = {"a": 3, "b": 2}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_ranking(["a", "a"], labels)


def test_unknown_candidate_is_rejected() -> None:
    labels = {"a": 3}
    with pytest.raises(ValueError, match="missing from benchmark"):
        evaluate_ranking(["b"], labels)


def test_metric_values_are_finite() -> None:
    labels = {"a": 0, "b": 0}
    metrics = evaluate_ranking(["a", "b"], labels, ks=(1, 2))
    assert metrics.reciprocal_rank == 0
    assert metrics.average_precision == 0
    assert all(math.isfinite(v) for v in metrics.ndcg_at_k.values())
