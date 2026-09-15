"""Grounded Gemini narrative evaluation with structured output.

The LLM is intentionally *not* allowed to set the final ranking score. HireFlow
keeps ranking deterministic and asks Gemini only to synthesize an explanation
from an identity-free evidence catalog. This prevents score drift and makes the
system easier to audit.
"""
from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hireflow.evaluation.base import recommendation_for_score
from hireflow.exceptions import EvaluationError
from hireflow.models import CandidateEvaluation, CandidateMatch, CandidateProfile, JobRequirements
from hireflow.ranking import build_evidence_catalog


class _StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroundedClaim(_StrictOutput):
    claim: str = Field(description="Concise job-relevant strength supported only by the supplied evidence IDs.")
    evidence_ids: list[str] = Field(min_length=1, max_length=3)


class GroundedNarrative(_StrictOutput):
    candidate_id: str
    strengths: list[GroundedClaim] = Field(default_factory=list, max_length=5)
    gap_requirements: list[str] = Field(default_factory=list, max_length=5)
    reasoning: str
    confidence: float = Field(ge=0, le=1)


def _build_prompt(
    profile: CandidateProfile,
    job: JobRequirements,
    match: CandidateMatch,
) -> tuple[str, dict[str, Any]]:
    catalog = build_evidence_catalog(profile)
    allowed_gaps = list(match.missing_requirements)
    payload = {
        "job": {
            "title": job.title,
            "experience_min_years": job.experience_min_years,
            "experience_max_years": job.experience_max_years,
            "responsibilities": job.responsibilities,
            "required_skills": job.required_skills,
            "required_software": job.required_software,
            "education_requirements": job.education_requirements,
            "preferred_skills": job.preferred_skills,
            "preferred_qualifications": job.preferred_qualifications,
            "preferred_certifications": job.preferred_certifications,
        },
        "deterministic_match": {
            "candidate_id": match.candidate_id,
            "score": match.final_score,
            "matched_requirements": match.matched_requirements,
            "missing_requirements": allowed_gaps,
            "component_scores": match.component_scores.model_dump() if match.component_scores else None,
        },
        "candidate_evidence": [
            {"evidence_id": item.evidence_id, "section": item.section, "text": item.text}
            for item in catalog
        ],
    }
    instructions = """
You are the explanation layer of a hiring decision-support system.

STRICT RULES:
1. Use only the supplied candidate_evidence. Never invent skills, employers, dates, certifications, or experience.
2. Candidate identity, name, contact details, age, gender, race, religion, disability, marital status and other protected/sensitive attributes are not provided and must never be inferred.
3. A missing requirement means only "direct evidence was not found in the supplied resume". Never state that the candidate definitely lacks a skill.
4. Every strength MUST cite 1-3 evidence_ids from candidate_evidence that directly support the claim.
5. gap_requirements may contain ONLY exact strings from deterministic_match.missing_requirements. Do not create new gaps.
6. Do not change or reinterpret the deterministic numeric score. You are writing the qualitative explanation only.
7. Keep reasoning concise, professional and recruiter-friendly. Treat this as decision support, not an autonomous hiring decision.

Return structured output matching the provided schema.
""".strip()
    return instructions + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2), payload


class GeminiGroundedEvaluator:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise EvaluationError("Gemini API key is required for Gemini evaluation")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise EvaluationError(
                "google-genai is not installed. Install project requirements to use Gemini evaluation."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _generate(self, prompt: str) -> GroundedNarrative:
        try:
            from google.genai import types
        except ImportError as exc:
            raise EvaluationError("google-genai is not installed") from exc

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._get_client().models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GroundedNarrative,
                        temperature=0.1,
                    ),
                )
                if isinstance(getattr(response, "parsed", None), GroundedNarrative):
                    return response.parsed
                if not getattr(response, "text", None):
                    raise EvaluationError("Gemini returned an empty response")
                return GroundedNarrative.model_validate_json(response.text)
            except Exception as exc:  # network/API/schema failures can be transient
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.initial_backoff_seconds * (2 ** (attempt - 1)))
        if isinstance(last_error, EvaluationError):
            raise last_error
        raise EvaluationError(
            f"Gemini evaluation failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def evaluate(
        self,
        profile: CandidateProfile,
        job: JobRequirements,
        match: CandidateMatch,
    ) -> CandidateEvaluation:
        prompt, _ = _build_prompt(profile, job, match)
        narrative = self._generate(prompt)
        if narrative.candidate_id != profile.candidate_id:
            raise EvaluationError("Gemini returned a mismatched candidate_id")

        catalog = {item.evidence_id: item for item in build_evidence_catalog(profile)}
        valid_gap_set = set(match.missing_requirements)
        strengths: list[str] = []
        used_ids: list[str] = []
        for claim in narrative.strengths:
            invalid_ids = [evidence_id for evidence_id in claim.evidence_ids if evidence_id not in catalog]
            if invalid_ids:
                raise EvaluationError(f"Gemini referenced unknown evidence IDs: {invalid_ids}")
            strengths.append(claim.claim)
            used_ids.extend(claim.evidence_ids)

        invalid_gaps = [gap for gap in narrative.gap_requirements if gap not in valid_gap_set]
        if invalid_gaps:
            raise EvaluationError(f"Gemini introduced unsupported gap requirements: {invalid_gaps}")

        # Resolve evidence text server-side so the model cannot fabricate quotations.
        resolved = []
        seen: set[str] = set()
        requirement_by_id = {item.evidence_id: item.requirement for item in match.evidence if item.evidence_id}
        for evidence_id in used_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            record = catalog[evidence_id]
            from hireflow.models import EvidenceItem
            resolved.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    requirement=requirement_by_id.get(evidence_id, "Supporting evidence"),
                    evidence=record.text,
                    source_section=record.section,
                )
            )

        if not strengths:
            strengths = [f"Evidence supports {item}." for item in match.matched_requirements[:5]]
        gaps = [f"No direct resume evidence found for {gap}." for gap in narrative.gap_requirements]
        if not gaps:
            gaps = [f"No direct resume evidence found for {gap}." for gap in match.missing_requirements[:5]]
        if not resolved:
            resolved = match.evidence[:10]

        score = float(match.final_score or 0.0)
        return CandidateEvaluation(
            candidate_id=profile.candidate_id,
            overall_score=score,
            recommendation=recommendation_for_score(score),
            strengths=strengths,
            gaps=gaps,
            matched_skills=match.matched_requirements,
            missing_skills=match.missing_requirements,
            evidence=resolved,
            reasoning=narrative.reasoning,
            confidence=narrative.confidence,
        )


__all__ = ["GeminiGroundedEvaluator", "GroundedClaim", "GroundedNarrative", "_build_prompt"]
