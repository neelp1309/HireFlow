"""Offline ranking metrics for retrieval and reranking experiments.

The module intentionally keeps ranking evaluation deterministic and independent
from any LLM call. Relevance labels are ordinal (0..3), while binary retrieval
metrics use a configurable threshold (default: label >= 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Metrics for a single ranked candidate list."""

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    strong_fit_rate_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    reciprocal_rank: float
    average_precision: float
    relevant_count: int
    strong_fit_count: int
    candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "precision_at_k": {str(k): round(v, 6) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): round(v, 6) for k, v in self.recall_at_k.items()},
            "strong_fit_rate_at_k": {
                str(k): round(v, 6) for k, v in self.strong_fit_rate_at_k.items()
            },
            "ndcg_at_k": {str(k): round(v, 6) for k, v in self.ndcg_at_k.items()},
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            "average_precision": round(self.average_precision, 6),
            "relevant_count": self.relevant_count,
            "strong_fit_count": self.strong_fit_count,
            "candidate_count": self.candidate_count,
        }


def _validate_ranking(ranked_ids: Sequence[str], labels: Mapping[str, int]) -> None:
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked_ids contains duplicate candidate IDs")
    unknown = [candidate_id for candidate_id in ranked_ids if candidate_id not in labels]
    if unknown:
        raise ValueError(f"Ranking contains candidate IDs missing from benchmark: {unknown[:5]}")
    invalid = {cid: label for cid, label in labels.items() if label < 0 or label > 3}
    if invalid:
        raise ValueError(f"Relevance labels must be in [0, 3]; invalid values: {invalid}")


def precision_at_k(
    ranked_ids: Sequence[str], labels: Mapping[str, int], k: int, relevant_threshold: int = 2
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_ranking(ranked_ids, labels)
    prefix = ranked_ids[:k]
    if not prefix:
        return 0.0
    relevant = sum(labels[candidate_id] >= relevant_threshold for candidate_id in prefix)
    return relevant / len(prefix)


def recall_at_k(
    ranked_ids: Sequence[str], labels: Mapping[str, int], k: int, relevant_threshold: int = 2
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_ranking(ranked_ids, labels)
    total_relevant = sum(label >= relevant_threshold for label in labels.values())
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(
        labels[candidate_id] >= relevant_threshold for candidate_id in ranked_ids[:k]
    )
    return retrieved_relevant / total_relevant


def strong_fit_rate_at_k(
    ranked_ids: Sequence[str], labels: Mapping[str, int], k: int, strong_label: int = 3
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_ranking(ranked_ids, labels)
    prefix = ranked_ids[:k]
    if not prefix:
        return 0.0
    strong = sum(labels[candidate_id] == strong_label for candidate_id in prefix)
    return strong / len(prefix)


def reciprocal_rank(
    ranked_ids: Sequence[str], labels: Mapping[str, int], relevant_threshold: int = 2
) -> float:
    _validate_ranking(ranked_ids, labels)
    for index, candidate_id in enumerate(ranked_ids, start=1):
        if labels[candidate_id] >= relevant_threshold:
            return 1.0 / index
    return 0.0


def average_precision(
    ranked_ids: Sequence[str], labels: Mapping[str, int], relevant_threshold: int = 2
) -> float:
    _validate_ranking(ranked_ids, labels)
    total_relevant = sum(label >= relevant_threshold for label in labels.values())
    if total_relevant == 0:
        return 0.0
    found = 0
    precision_sum = 0.0
    for index, candidate_id in enumerate(ranked_ids, start=1):
        if labels[candidate_id] >= relevant_threshold:
            found += 1
            precision_sum += found / index
    return precision_sum / total_relevant


def dcg_at_k(ranked_ids: Sequence[str], labels: Mapping[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_ranking(ranked_ids, labels)
    score = 0.0
    for index, candidate_id in enumerate(ranked_ids[:k], start=1):
        gain = (2 ** labels[candidate_id]) - 1
        score += gain / log2(index + 1)
    return score


def ndcg_at_k(ranked_ids: Sequence[str], labels: Mapping[str, int], k: int) -> float:
    actual = dcg_at_k(ranked_ids, labels, k)
    ideal_labels = sorted(labels.values(), reverse=True)[:k]
    ideal = sum(((2**label) - 1) / log2(index + 1) for index, label in enumerate(ideal_labels, start=1))
    return actual / ideal if ideal else 0.0


def evaluate_ranking(
    ranked_ids: Sequence[str],
    labels: Mapping[str, int],
    ks: Iterable[int] = (5, 10, 20),
    relevant_threshold: int = 2,
) -> RankingMetrics:
    """Evaluate one complete or partial ranking against ordinal benchmark labels."""
    _validate_ranking(ranked_ids, labels)
    ks_tuple = tuple(sorted(set(ks)))
    if not ks_tuple or any(k <= 0 for k in ks_tuple):
        raise ValueError("ks must contain positive integers")

    return RankingMetrics(
        precision_at_k={
            k: precision_at_k(ranked_ids, labels, k, relevant_threshold) for k in ks_tuple
        },
        recall_at_k={k: recall_at_k(ranked_ids, labels, k, relevant_threshold) for k in ks_tuple},
        strong_fit_rate_at_k={k: strong_fit_rate_at_k(ranked_ids, labels, k) for k in ks_tuple},
        ndcg_at_k={k: ndcg_at_k(ranked_ids, labels, k) for k in ks_tuple},
        reciprocal_rank=reciprocal_rank(ranked_ids, labels, relevant_threshold),
        average_precision=average_precision(ranked_ids, labels, relevant_threshold),
        relevant_count=sum(label >= relevant_threshold for label in labels.values()),
        strong_fit_count=sum(label == 3 for label in labels.values()),
        candidate_count=len(labels),
    )
