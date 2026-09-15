"""Deterministic requirement matching with traceable evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz.fuzz import partial_ratio

from hireflow.models import CandidateProfile, JobRequirements
from hireflow.ranking.evidence import EvidenceRecord, build_evidence_catalog, record_to_evidence
from hireflow.ranking.taxonomy import Concept, MatchingTaxonomy

_TOKEN_RE = re.compile(r"[^a-z0-9+#.]+")


def normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.sub(" ", text.casefold()).split())


def _alias_in_text(alias: str, text: str) -> bool:
    a = normalize(alias)
    t = normalize(text)
    if not a or not t:
        return False
    if a in t:
        return True
    # Fuzzy matching is a fallback for minor PDF/tokenization variations only.
    return len(a) >= 8 and partial_ratio(a, t) >= 92


def _concepts_present(texts: list[str], concepts: tuple[Concept, ...]) -> list[Concept]:
    joined = "\n".join(texts)
    return [concept for concept in concepts if any(_alias_in_text(alias, joined) for alias in concept.aliases)]


def _best_evidence(concept: Concept, catalog: list[EvidenceRecord]) -> EvidenceRecord | None:
    # Prefer more structured/direct sections over the generic summary.
    priority = {
        "technical_skills": 0,
        "software": 1,
        "certifications": 2,
        "work_experience": 3,
        "education": 4,
        "headline": 5,
        "experience": 6,
        "professional_summary": 7,
    }
    matches = [r for r in catalog if any(_alias_in_text(alias, r.text) for alias in concept.aliases)]
    return min(matches, key=lambda r: priority.get(r.section, 99), default=None)


@dataclass(frozen=True, slots=True)
class CoverageResult:
    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    evidence: tuple


class RequirementMatcher:
    def __init__(self, taxonomy: MatchingTaxonomy) -> None:
        self.taxonomy = taxonomy

    def required_concepts(self, job: JobRequirements) -> list[Concept]:
        texts = [*job.responsibilities, *job.required_skills]
        return _concepts_present(texts, self.taxonomy.concepts)

    def concept_coverage(self, profile: CandidateProfile, job: JobRequirements) -> CoverageResult:
        required = self.required_concepts(job)
        if not required:
            return CoverageResult(100.0, (), (), ())

        catalog = build_evidence_catalog(profile)
        matched: list[str] = []
        missing: list[str] = []
        evidence = []
        total_weight = sum(c.weight for c in required)
        matched_weight = 0.0

        for concept in required:
            record = _best_evidence(concept, catalog)
            if record:
                matched.append(concept.label)
                matched_weight += concept.weight
                evidence.append(record_to_evidence(record, concept.label))
            else:
                missing.append(concept.label)

        score = 100.0 * matched_weight / total_weight if total_weight else 100.0
        return CoverageResult(score, tuple(matched), tuple(missing), tuple(evidence))

    def software_coverage(self, profile: CandidateProfile, job: JobRequirements) -> CoverageResult:
        catalog = build_evidence_catalog(profile)
        candidate_software = "\n".join(profile.software + [c.name for c in profile.certifications])
        requested = "\n".join(job.required_software)
        applicable = [
            group
            for group in self.taxonomy.software_groups
            if any(_alias_in_text(alias, requested) for alias in group.aliases)
            or group.key == "accounting_software" and "accounting software" in normalize(requested)
        ]
        if not applicable:
            return CoverageResult(100.0, (), (), ())

        matched: list[str] = []
        missing: list[str] = []
        evidence = []
        for group in applicable:
            aliases = group.aliases
            hit_alias = next((alias for alias in aliases if _alias_in_text(alias, candidate_software)), None)
            if hit_alias:
                matched.append(group.label)
                concept = Concept(group.key, group.label, aliases)
                record = _best_evidence(concept, catalog)
                if record:
                    evidence.append(record_to_evidence(record, group.label))
            else:
                missing.append(group.label)
        return CoverageResult(100.0 * len(matched) / len(applicable), tuple(matched), tuple(missing), tuple(evidence))

    def education_score(self, profile: CandidateProfile, job: JobRequirements) -> CoverageResult:
        if not job.education_requirements:
            return CoverageResult(100.0, (), (), ())
        if not profile.education:
            return CoverageResult(0.0, (), ("Relevant bachelor's degree",), ())

        catalog = build_evidence_catalog(profile)
        best_score = 0.0
        best_label = ""
        best_record: EvidenceRecord | None = None
        for edu in profile.education:
            text = normalize(f"{edu.degree} {edu.field_of_study or ''}")
            if any(field in text for field in self.taxonomy.strong_education_fields):
                score = 100.0
            elif any(field in text for field in self.taxonomy.related_education_fields):
                score = 85.0
            elif any(token in text for token in ("bachelor", "master", "mba", "degree")):
                score = 65.0
            else:
                score = 40.0
            if score > best_score:
                best_score = score
                best_label = f"Relevant education ({edu.degree}{' in ' + edu.field_of_study if edu.field_of_study else ''})"
                best_record = next((r for r in catalog if r.section == "education" and normalize(edu.degree) in normalize(r.text)), None)

        if best_score >= 65:
            ev = (record_to_evidence(best_record, "Relevant education"),) if best_record else ()
            return CoverageResult(best_score, (best_label,), (), ev)
        return CoverageResult(best_score, (), ("Relevant bachelor's degree",), ())

    def preferred_coverage(self, profile: CandidateProfile, job: JobRequirements) -> CoverageResult:
        # Only score criteria for which the resume can reasonably provide direct evidence.
        job_text = "\n".join([*job.preferred_skills, *job.preferred_qualifications, *job.preferred_certifications])
        applicable = _concepts_present([job_text], self.taxonomy.preferred_criteria)
        if not applicable:
            return CoverageResult(50.0, (), (), ())

        catalog = build_evidence_catalog(profile)
        matched: list[str] = []
        missing: list[str] = []
        evidence = []
        for concept in applicable:
            record = _best_evidence(concept, catalog)
            if record:
                matched.append(concept.label)
                evidence.append(record_to_evidence(record, concept.label))
            else:
                missing.append(concept.label)
        return CoverageResult(100.0 * len(matched) / len(applicable), tuple(matched), tuple(missing), tuple(evidence))
