#!/usr/bin/env python
"""Run Phase 4 retrieval/ranking experiments against the silver benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.config import get_settings  # noqa: E402
from hireflow.evaluation import (  # noqa: E402
    benchmark_labels,
    benchmark_summary,
    evaluate_ranking,
    load_benchmark,
    run_hybrid_experiment,
    run_retrieval_experiment,
)
from hireflow.models import CandidateProfile, JobRequirements  # noqa: E402
from hireflow.ranking import ScoringWeights  # noqa: E402
from hireflow.utils import read_json, read_jsonl  # noqa: E402


def _random_baseline(labels: dict[str, int], ks: tuple[int, ...], threshold: int, trials: int = 1000) -> dict[str, object]:
    rng = random.Random(42)
    ids = list(labels)
    metrics = []
    for _ in range(trials):
        ranking = ids.copy()
        rng.shuffle(ranking)
        metrics.append(evaluate_ranking(ranking, labels, ks=ks, relevant_threshold=threshold))

    def avg(values):
        return sum(values) / len(values)

    return {
        "trials": trials,
        "precision_at_k": {str(k): round(avg([m.precision_at_k[k] for m in metrics]), 6) for k in ks},
        "recall_at_k": {str(k): round(avg([m.recall_at_k[k] for m in metrics]), 6) for k in ks},
        "strong_fit_rate_at_k": {
            str(k): round(avg([m.strong_fit_rate_at_k[k] for m in metrics]), 6) for k in ks
        },
        "ndcg_at_k": {str(k): round(avg([m.ndcg_at_k[k] for m in metrics]), 6) for k in ks},
        "reciprocal_rank": round(avg([m.reciprocal_rank for m in metrics]), 6),
        "average_precision": round(avg([m.average_precision for m in metrics]), 6),
    }


def _metric(experiment: dict[str, object], group: str, k: int) -> float:
    return float(experiment["metrics"][group][str(k)])  # type: ignore[index]


def _render_markdown(payload: dict[str, object]) -> str:
    exps = payload["experiments"]  # type: ignore[assignment]
    lines = [
        "# Phase 4 Experiment Report",
        "",
        "## Benchmark",
        "",
        f"- Candidates: **{payload['benchmark']['candidate_count']}**",  # type: ignore[index]
        f"- Binary relevance threshold: **label >= {payload['relevant_threshold']}**",
        "- Labels: 0=not suitable, 1=weak, 2=moderate, 3=strong",
        "- Status: **project-authored silver benchmark; independent human review still required**",
        "",
        "## Ranking results",
        "",
        "| Experiment | P@5 | P@10 | Strong@5 | Strong@10 | NDCG@5 | NDCG@10 | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in exps:
        lines.append(
            "| {name} | {p5:.3f} | {p10:.3f} | {s5:.3f} | {s10:.3f} | {n5:.3f} | {n10:.3f} | {ap:.3f} |".format(
                name=exp["experiment_name"],
                p5=_metric(exp, "precision_at_k", 5),
                p10=_metric(exp, "precision_at_k", 10),
                s5=_metric(exp, "strong_fit_rate_at_k", 5),
                s10=_metric(exp, "strong_fit_rate_at_k", 10),
                n5=_metric(exp, "ndcg_at_k", 5),
                n10=_metric(exp, "ndcg_at_k", 10),
                ap=float(exp["metrics"]["average_precision"]),
            )
        )
    random_metrics = payload["random_baseline"]  # type: ignore[assignment]
    lines.append(
        "| Random baseline (mean) | {p5:.3f} | {p10:.3f} | {s5:.3f} | {s10:.3f} | {n5:.3f} | {n10:.3f} | {ap:.3f} |".format(
            p5=float(random_metrics["precision_at_k"]["5"]),
            p10=float(random_metrics["precision_at_k"]["10"]),
            s5=float(random_metrics["strong_fit_rate_at_k"]["5"]),
            s10=float(random_metrics["strong_fit_rate_at_k"]["10"]),
            n5=float(random_metrics["ndcg_at_k"]["5"]),
            n10=float(random_metrics["ndcg_at_k"]["10"]),
            ap=float(random_metrics["average_precision"]),
        )
    )
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- There is only **one supplied job description**, so these metrics are descriptive for this JD and are not a generalization estimate.",
        "- The benchmark is a **silver benchmark**, authored from an explicit rubric and not external human ground truth.",
        "- Hybrid weights were **not optimized against these labels**. Tuning on this single query would overfit the benchmark.",
        "- Gemini dense-embedding experiments are supported by the codebase but are not executed without a configured API key.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, default=PROJECT_ROOT / "data" / "processed" / "job.json")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "data" / "processed" / "candidates.jsonl")
    parser.add_argument("--benchmark", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "senior_accountant_benchmark.csv")
    parser.add_argument("--evaluation-config", type=Path, default=PROJECT_ROOT / "configs" / "evaluation.yaml")
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "matching_taxonomy.yaml")
    parser.add_argument("--full-index", type=Path, default=PROJECT_ROOT / "artifacts" / "indexes" / "baseline_full_tfidf")
    parser.add_argument("--section-index", type=Path, default=PROJECT_ROOT / "artifacts" / "indexes" / "section_tfidf")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "evaluation")
    args = parser.parse_args()

    config = yaml.safe_load(args.evaluation_config.read_text(encoding="utf-8"))
    ks = tuple(int(k) for k in config["metrics"]["k_values"])
    threshold = int(config["benchmark"]["relevant_threshold"])

    job = read_json(args.job, JobRequirements)
    candidates_list = read_jsonl(args.candidates, CandidateProfile)
    candidates = {candidate.candidate_id: candidate for candidate in candidates_list}
    records = load_benchmark(args.benchmark, expected_job_id=job.job_id)
    labels = benchmark_labels(records)

    if set(labels) != set(candidates):
        missing_labels = sorted(set(candidates) - set(labels))
        unknown_labels = sorted(set(labels) - set(candidates))
        raise ValueError(
            f"Benchmark/candidate mismatch. Missing labels={missing_labels}, unknown labels={unknown_labels}"
        )

    weights = ScoringWeights(
        semantic=settings.weight_semantic,
        required_skills=settings.weight_required_skills,
        experience=settings.weight_experience,
        role=settings.weight_role,
        preferred=settings.weight_preferred,
    )

    experiments = [
        run_retrieval_experiment(
            "TF-IDF full-resume retrieval",
            job,
            args.full_index,
            labels,
            ks,
            threshold,
        ),
        run_retrieval_experiment(
            "TF-IDF section-aware retrieval",
            job,
            args.section_index,
            labels,
            ks,
            threshold,
        ),
        run_hybrid_experiment(
            "TF-IDF full-resume + hybrid reranking",
            job,
            candidates,
            args.full_index,
            args.taxonomy,
            labels,
            weights,
            ks,
            threshold,
        ),
        run_hybrid_experiment(
            "TF-IDF section-aware + hybrid reranking",
            job,
            candidates,
            args.section_index,
            args.taxonomy,
            labels,
            weights,
            ks,
            threshold,
        ),
    ]

    payload = {
        "phase": 4,
        "job_id": job.job_id,
        "job_title": job.title,
        "relevant_threshold": threshold,
        "k_values": list(ks),
        "benchmark": benchmark_summary(records),
        "weights": {
            "semantic": weights.semantic,
            "required_skills": weights.required_skills,
            "experience": weights.experience,
            "role": weights.role,
            "preferred": weights.preferred,
        },
        "experiments": [experiment.to_dict() for experiment in experiments],
        "random_baseline": _random_baseline(labels, ks, threshold),
        "limitations": [
            "Single-JD benchmark; no claim of cross-role generalization.",
            "Project-authored silver labels require independent human review for production claims.",
            "Hybrid weights intentionally not tuned on this benchmark to avoid one-query overfitting.",
            "Gemini dense embedding experiment not executed without an API key.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "phase4_experiment_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "phase4_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "p@5",
                "p@10",
                "r@5",
                "r@10",
                "strong@5",
                "strong@10",
                "ndcg@5",
                "ndcg@10",
                "mrr",
                "average_precision",
            ]
        )
        for exp in payload["experiments"]:
            writer.writerow(
                [
                    exp["experiment_name"],
                    _metric(exp, "precision_at_k", 5),
                    _metric(exp, "precision_at_k", 10),
                    _metric(exp, "recall_at_k", 5),
                    _metric(exp, "recall_at_k", 10),
                    _metric(exp, "strong_fit_rate_at_k", 5),
                    _metric(exp, "strong_fit_rate_at_k", 10),
                    _metric(exp, "ndcg_at_k", 5),
                    _metric(exp, "ndcg_at_k", 10),
                    exp["metrics"]["reciprocal_rank"],
                    exp["metrics"]["average_precision"],
                ]
            )

    comparison_path = args.output_dir / "phase4_ranking_comparison.csv"
    experiment_names = [exp["experiment_name"] for exp in payload["experiments"]]
    ranking_maps = {
        exp["experiment_name"]: {candidate_id: rank for rank, candidate_id in enumerate(exp["ranked_candidate_ids"], start=1)}
        for exp in payload["experiments"]
    }
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "relevance_label", *experiment_names])
        for candidate_id in sorted(labels):
            writer.writerow(
                [candidate_id, labels[candidate_id], *[ranking_maps[name][candidate_id] for name in experiment_names]]
            )

    md_path = args.output_dir / "phase4_experiment_report.md"
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "results": str(json_path),
                "metrics": str(csv_path),
                "ranking_comparison": str(comparison_path),
                "report": str(md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
