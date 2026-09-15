"""Requirement-aware hybrid reranking for retrieved candidates."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from hireflow.models import CandidateMatch, CandidateProfile, CandidateRetrievalResult, JobRequirements, MatchComponentScores
from hireflow.ranking.matcher import RequirementMatcher, normalize


def semantic_score(retrieval_score: float) -> float:
    """Calibrate non-negative cosine similarity to a readable 0-100 component.

    Square-root calibration is deliberately conservative for the low-valued
    TF-IDF baseline while remaining monotonic for dense cosine similarities.
    Phase 4 will test/calibrate this choice against relevance labels.
    """
    return max(0.0, min(100.0, sqrt(max(0.0, retrieval_score)) * 100.0))


def _seniority_level(title: str | None) -> int:
    text = normalize(title or "")
    if any(x in text for x in ("intern", "recent graduate", "graduate")):
        return 0
    if any(x in text for x in ("assistant", "junior", "associate")):
        return 1
    if any(x in text for x in ("director", "vp", "vice president", "cfo", "chief")):
        return 5
    if any(x in text for x in ("manager", "controller", "lead", "supervisor", "principal")):
        return 4
    if "senior" in text:
        return 3
    if any(x in text for x in ("staff", "accountant", "analyst", "specialist")):
        return 2
    return 2


def _role_family(title: str | None) -> str:
    text = normalize(title or "")
    if any(x in text for x in ("accountant", "accounting", "controller", "audit", "tax")):
        return "accounting"
    if any(x in text for x in ("financial analyst", "finance analyst", "finance")):
        return "finance"
    return "other"


def role_alignment(candidate_title: str | None, job_title: str) -> float:
    cand = normalize(candidate_title or "")
    job = normalize(job_title)
    if not cand:
        return 30.0
    if cand == job:
        return 100.0

    # Normalize accountant/accounting morphology for core-role overlap.
    def stemmed(text: str) -> set[str]:
        out: set[str] = set()
        for token in text.split():
            if token in {"accounting", "accountant"}:
                out.add("account")
            elif token not in {"senior", "junior", "staff", "assistant", "lead"}:
                out.add(token)
        return out

    job_core = stemmed(job)
    cand_core = stemmed(cand)
    if job_core and job_core.intersection(cand_core):
        family_score = 100.0
    elif _role_family(candidate_title) == _role_family(job_title):
        family_score = 60.0
    elif {_role_family(candidate_title), _role_family(job_title)} == {"accounting", "finance"}:
        family_score = 58.0
    else:
        family_score = 35.0

    level_gap = abs(_seniority_level(candidate_title) - _seniority_level(job_title))
    seniority_score = max(40.0, 100.0 - 12.0 * level_gap)
    return round(0.78 * family_score + 0.22 * seniority_score, 2)


def experience_alignment(years: float | None, minimum: float | None, maximum: float | None) -> float:
    if minimum is None and maximum is None:
        return 100.0
    if years is None:
        return 35.0
    if minimum is not None and years < minimum:
        return max(10.0, 100.0 - 45.0 * (minimum - years))
    if maximum is not None and years > maximum:
        # Overqualification is a softer mismatch than insufficient experience.
        return max(45.0, 100.0 - 12.0 * (years - maximum))
    return 100.0


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    semantic: float = 0.35
    required_skills: float = 0.30
    experience: float = 0.15
    role: float = 0.10
    preferred: float = 0.10

    def __post_init__(self) -> None:
        total = self.semantic + self.required_skills + self.experience + self.role + self.preferred
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0; got {total:.6f}")


class HybridCandidateReranker:
    """Rerank semantic candidates using explicit, explainable job-fit features."""

    def __init__(self, matcher: RequirementMatcher, weights: ScoringWeights | None = None) -> None:
        self.matcher = matcher
        self.weights = weights or ScoringWeights()

    def score_candidate(
        self,
        retrieval: CandidateRetrievalResult,
        profile: CandidateProfile,
        job: JobRequirements,
    ) -> CandidateMatch:
        concept = self.matcher.concept_coverage(profile, job)
        software = self.matcher.software_coverage(profile, job)
        education = self.matcher.education_score(profile, job)
        preferred = self.matcher.preferred_coverage(profile, job)

        required_score = 0.70 * concept.score + 0.20 * software.score + 0.10 * education.score
        components = MatchComponentScores(
            semantic=semantic_score(retrieval.retrieval_score),
            required_skills=required_score,
            experience=experience_alignment(profile.total_experience_years, job.experience_min_years, job.experience_max_years),
            role=role_alignment(profile.headline, job.title),
            preferred=preferred.score,
        )
        w = self.weights
        final_score = (
            w.semantic * components.semantic
            + w.required_skills * components.required_skills
            + w.experience * components.experience
            + w.role * components.role
            + w.preferred * components.preferred
        )

        matched = [*concept.matched, *software.matched, *education.matched]
        missing = [*concept.missing, *software.missing, *education.missing]
        evidence = [*concept.evidence, *software.evidence, *education.evidence, *preferred.evidence]
        # Deduplicate evidence by ID while retaining first occurrence.
        deduped = []
        seen_ids: set[str] = set()
        for item in evidence:
            key = item.evidence_id or f"{item.source_section}:{item.evidence}"
            if key not in seen_ids:
                seen_ids.add(key)
                deduped.append(item)

        return CandidateMatch(
            candidate_id=profile.candidate_id,
            retrieval_score=retrieval.retrieval_score,
            component_scores=components,
            final_score=round(final_score, 2),
            matched_requirements=list(dict.fromkeys(matched)),
            missing_requirements=list(dict.fromkeys(missing)),
            evidence=deduped,
        )

    def rerank(
        self,
        retrieval_results: list[CandidateRetrievalResult],
        candidates: dict[str, CandidateProfile],
        job: JobRequirements,
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        matches = [
            self.score_candidate(result, candidates[result.candidate_id], job)
            for result in retrieval_results
            if result.candidate_id in candidates
        ]
        matches.sort(key=lambda item: item.final_score or 0.0, reverse=True)
        if top_k is not None:
            matches = matches[:top_k]
        return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(matches, start=1)]
