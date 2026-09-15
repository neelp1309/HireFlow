"""Evaluate a reranked shortlist with optional Gemini and safe offline fallback."""
from __future__ import annotations

import logging

from hireflow.evaluation.base import CandidateEvaluator
from hireflow.evaluation.heuristic import HeuristicGroundedEvaluator
from hireflow.exceptions import EvaluationError
from hireflow.models import CandidateEvaluation, CandidateMatch, CandidateProfile, JobRequirements

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(self, evaluator: CandidateEvaluator, fallback: CandidateEvaluator | None = None) -> None:
        self.evaluator = evaluator
        self.fallback = fallback or HeuristicGroundedEvaluator()

    def evaluate_shortlist(
        self,
        matches: list[CandidateMatch],
        candidates: dict[str, CandidateProfile],
        job: JobRequirements,
    ) -> list[CandidateEvaluation]:
        results: list[CandidateEvaluation] = []
        for match in matches:
            profile = candidates[match.candidate_id]
            try:
                results.append(self.evaluator.evaluate(profile, job, match))
            except EvaluationError as exc:
                logger.warning("Evaluator failed for %s; using grounded fallback: %s", match.candidate_id, exc)
                results.append(self.fallback.evaluate(profile, job, match))
        return results
