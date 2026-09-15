from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hireflow.evaluation.benchmark import benchmark_labels, benchmark_summary, load_benchmark


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["job_id", "candidate_id", "relevance_label", "label_name", "rationale", "review_status"]
        )
        writer.writerows(rows)


def test_benchmark_load_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.csv"
    _write(
        path,
        [
            ["j1", "c1", "3", "strong_fit", "good", "silver"],
            ["j1", "c2", "1", "weak_fit", "gap", "silver"],
        ],
    )
    records = load_benchmark(path, expected_job_id="j1")
    assert benchmark_labels(records) == {"c1": 3, "c2": 1}
    summary = benchmark_summary(records)
    assert summary["candidate_count"] == 2
    assert summary["label_distribution"]["strong_fit"] == 1


def test_benchmark_rejects_label_name_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.csv"
    _write(path, [["j1", "c1", "3", "weak_fit", "bad", "silver"]])
    with pytest.raises(ValueError, match="Label name mismatch"):
        load_benchmark(path)
