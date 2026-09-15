# Responsible AI and Hiring Boundary

Hiring is a high-impact domain. HireFlow is intentionally framed as **recruiter decision support**, not an autonomous employment decision system.

## Implemented safeguards

### Identity separation

Candidate name, email, phone, LinkedIn and location can be retained for recruiter display but are excluded from:

- retrieval matching text;
- hybrid ranking features;
- benchmark relevance criteria;
- deterministic evidence matching;
- Gemini explanation prompts.

### Evidence-first explanations

HireFlow distinguishes between absence of resume evidence and absence of ability. It reports “direct evidence not found” rather than inferring that a candidate lacks a skill.

### Deterministic score ownership

The LLM does not set the numeric fit score. The score is computed from explicit, configurable components. This improves repeatability and auditability.

### Public API minimization

API responses intentionally omit raw resume text, full names and contact details.

## What this project does NOT prove

The current evaluation does not establish fairness, absence of disparate impact, legal compliance, or generalization across occupations. The benchmark contains one job description and project-authored labels.

## Production requirements beyond this portfolio project

A real deployment should include, at minimum:

- legal and HR policy review;
- documented human oversight;
- explicit prohibited-feature policy;
- subgroup/fairness evaluation where legally and ethically appropriate;
- accessibility review;
- candidate data retention and deletion controls;
- encryption at rest and in transit;
- role-based access control and tenant isolation;
- audit logs and model/rule versioning;
- incident response and appeal/review procedures;
- independent validation of ranking labels and thresholds.

The system should support human judgment, not silently replace it.
