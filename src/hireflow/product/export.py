"""Export product results without exposing hidden ranking internals or secrets."""
from __future__ import annotations

import csv
import io
import json

from hireflow.product.models import ProductSearchBundle


def results_to_csv(bundle: ProductSearchBundle) -> str:
    buffer = io.StringIO()
    fields = [
        "rank",
        "candidate_id",
        "candidate_name",
        "headline",
        "experience_years",
        "score",
        "recommendation",
        "semantic_score",
        "requirements_score",
        "experience_score",
        "role_score",
        "preferred_score",
        "matched_requirements",
        "missing_requirements",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for item in bundle.results:
        comp = item.match.component_scores
        writer.writerow(
            {
                "rank": item.match.rank,
                "candidate_id": item.profile.candidate_id,
                "candidate_name": item.profile.full_name or "",
                "headline": item.profile.headline or "",
                "experience_years": item.profile.total_experience_years or "",
                "score": item.match.final_score,
                "recommendation": item.evaluation.recommendation.value,
                "semantic_score": comp.semantic if comp else "",
                "requirements_score": comp.required_skills if comp else "",
                "experience_score": comp.experience if comp else "",
                "role_score": comp.role if comp else "",
                "preferred_score": comp.preferred if comp else "",
                "matched_requirements": " | ".join(item.match.matched_requirements),
                "missing_requirements": " | ".join(item.match.missing_requirements),
            }
        )
    return buffer.getvalue()


def results_to_json(bundle: ProductSearchBundle) -> str:
    """Return recruiter-facing JSON; contact details are intentionally omitted."""
    payload = {
        "job": {
            "job_id": bundle.job.job_id,
            "title": bundle.job.title,
            "company": bundle.job.company,
            "experience_min_years": bundle.job.experience_min_years,
            "experience_max_years": bundle.job.experience_max_years,
        },
        "run": {
            "candidate_count": bundle.candidate_count,
            "processing_time_ms": bundle.processing_time_ms,
            "embedding_model": bundle.embedding_model,
            "vector_backend": bundle.vector_backend,
            "evaluator_used": bundle.evaluator_used,
            "config": bundle.config.model_dump(mode="json"),
        },
        "results": [
            {
                "rank": item.match.rank,
                "candidate_id": item.profile.candidate_id,
                "candidate_name": item.profile.full_name,
                "headline": item.profile.headline,
                "experience_years": item.profile.total_experience_years,
                "score": item.match.final_score,
                "recommendation": item.evaluation.recommendation.value,
                "component_scores": item.match.component_scores.model_dump(mode="json")
                if item.match.component_scores
                else None,
                "strengths": item.evaluation.strengths,
                "gaps": item.evaluation.gaps,
                "matched_requirements": item.match.matched_requirements,
                "missing_requirements": item.match.missing_requirements,
                "evidence": [e.model_dump(mode="json") for e in item.evaluation.evidence],
                "reasoning": item.evaluation.reasoning,
                "confidence": item.evaluation.confidence,
            }
            for item in bundle.results
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
