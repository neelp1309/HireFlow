"""Benchmark loading and validation utilities."""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    job_id: str
    candidate_id: str
    relevance_label: int
    label_name: str
    rationale: str
    review_status: str


LABEL_NAMES = {0: "not_suitable", 1: "weak_fit", 2: "moderate_fit", 3: "strong_fit"}


def load_benchmark(path: str | Path, expected_job_id: str | None = None) -> list[BenchmarkRecord]:
    source = Path(path)
    records: list[BenchmarkRecord] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "job_id",
            "candidate_id",
            "relevance_label",
            "label_name",
            "rationale",
            "review_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Benchmark is missing columns: {sorted(missing)}")
        for row in reader:
            label = int(row["relevance_label"])
            if label not in LABEL_NAMES:
                raise ValueError(f"Invalid relevance label for {row['candidate_id']}: {label}")
            if row["label_name"] != LABEL_NAMES[label]:
                raise ValueError(
                    f"Label name mismatch for {row['candidate_id']}: expected {LABEL_NAMES[label]}"
                )
            if expected_job_id is not None and row["job_id"] != expected_job_id:
                raise ValueError(
                    f"Unexpected job_id for {row['candidate_id']}: {row['job_id']} != {expected_job_id}"
                )
            records.append(
                BenchmarkRecord(
                    job_id=row["job_id"],
                    candidate_id=row["candidate_id"],
                    relevance_label=label,
                    label_name=row["label_name"],
                    rationale=row["rationale"],
                    review_status=row["review_status"],
                )
            )

    candidate_ids = [record.candidate_id for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Benchmark contains duplicate candidate IDs")
    return records


def benchmark_labels(records: list[BenchmarkRecord]) -> dict[str, int]:
    return {record.candidate_id: record.relevance_label for record in records}


def benchmark_summary(records: list[BenchmarkRecord]) -> dict[str, object]:
    counts = Counter(record.relevance_label for record in records)
    review_counts = Counter(record.review_status for record in records)
    return {
        "candidate_count": len(records),
        "label_distribution": {
            LABEL_NAMES[label]: counts.get(label, 0) for label in sorted(LABEL_NAMES)
        },
        "review_status_distribution": dict(sorted(review_counts.items())),
    }
