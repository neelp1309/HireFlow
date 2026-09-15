"""Offline grounded evaluator used for tests, demos and LLM fallback."""
from __future__ import annotations

from hireflow.evaluation.base import recommendation_for_score
from hireflow.models import CandidateEvaluation, CandidateMatch, CandidateProfile, JobRequirements


class HeuristicGroundedEvaluator:
    """Produce an evidence-backed evaluation without an external LLM call."""

    def evaluate(
        self,
        profile: CandidateProfile,
        job: JobRequirements,
        match: CandidateMatch,
    ) -> CandidateEvaluation:
        score = float(match.final_score or 0.0)
        strengths = [f"Evidence supports {item}." for item in match.matched_requirements[:5]]
        gaps = [f"No direct resume evidence found for {item}." for item in match.missing_requirements[:5]]
        evidence = match.evidence[:10]

        evidence_ratio = min(1.0, len(evidence) / max(1, len(match.matched_requirements)))
        confidence = round(0.55 + 0.35 * evidence_ratio, 2)
        role = profile.headline or "candidate profile"
        reasoning = (
            f"The {role} profile received a deterministic job-fit score of {score:.1f}/100. "
            f"The assessment is based on explicit role, experience, requirement, software, education, "
            f"preferred-criteria and retrieval signals. Missing items mean only that direct evidence was "
            f"not found in the supplied resume; they do not prove the candidate lacks that capability."
        )

        return CandidateEvaluation(
            candidate_id=profile.candidate_id,
            overall_score=score,
            recommendation=recommendation_for_score(score),
            strengths=strengths,
            gaps=gaps,
            matched_skills=match.matched_requirements,
            missing_skills=match.missing_requirements,
            evidence=evidence,
            reasoning=reasoning,
            confidence=confidence,
        )
