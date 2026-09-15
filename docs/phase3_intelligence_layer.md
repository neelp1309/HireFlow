# Phase 3 — Intelligence Layer

Phase 3 converts HireFlow from a semantic-search demo into an explainable candidate-ranking system.

## Implemented flow

```text
Semantic retrieval
      ↓
Top 20 candidate profiles
      ↓
Requirement-aware feature extraction
      ├── calibrated semantic relevance
      ├── required accounting concept coverage
      ├── required software coverage
      ├── education alignment
      ├── experience alignment
      ├── role/seniority alignment
      └── preferred-criteria coverage
      ↓
Hybrid deterministic reranker
      ↓
Top 10 candidates
      ↓
Grounded evaluation
      ├── deterministic score remains authoritative
      ├── identity-free evidence catalog
      ├── Gemini structured narrative (optional)
      └── offline evidence-backed fallback
      ↓
Score + strengths + gaps + cited resume evidence
```

## Why the LLM does not set the ranking score

A recruiter-facing score should not drift merely because an LLM was sampled twice. HireFlow therefore separates:

- **ranking**: deterministic and inspectable;
- **explanation**: generative, but constrained to supplied evidence.

The final score is calculated before Gemini is called. Gemini is only asked to explain the shortlist.

## Hybrid score

The Phase 3 baseline weights are:

| Component | Weight |
|---|---:|
| Semantic relevance | 25% |
| Required requirements | 30% |
| Experience alignment | 15% |
| Role/seniority alignment | 20% |
| Preferred criteria | 10% |

These are still hypotheses. Phase 4 will evaluate/tune them against a labeled relevance benchmark.

### Required-requirements component

The required component is itself composed of:

```text
70% accounting concept coverage
20% required software coverage
10% education alignment
```

The accounting concept taxonomy lives in `configs/matching_taxonomy.yaml`, making the matching logic auditable and replaceable rather than hidden inside a prompt.

## Explainable concept matching

Verbose JD requirements are mapped into concepts such as:

- financial reporting;
- month/year-end close;
- journal entries;
- account reconciliation;
- budgeting;
- variance analysis;
- payroll;
- audit;
- tax preparation/compliance;
- financial modeling and forecasting;
- AP/AR;
- GAAP;
- collaboration/mentoring/process improvement.

Each match is linked to an identity-free evidence item from structured resume sections. Missing requirements are reported as **"direct evidence was not found"**, never as proof that the candidate lacks a capability.

## Experience alignment

The supplied JD asks for 3–5 years. Phase 3 treats insufficient experience more strongly than overqualification:

```text
3–5 years  -> 100
2 years    -> 55
1 year     -> 10
6 years    -> 88
7 years    -> 76
```

This is an explicit business hypothesis and is eligible for later tuning.

## Role alignment

Role scoring combines:

1. core title overlap (e.g. Accountant ↔ Senior Accountant),
2. broad role family (accounting vs finance vs other), and
3. seniority distance.

This prevents an adjacent role from winning simply because it shares many accounting keywords.

## Grounded Gemini evaluator

`GeminiGroundedEvaluator` uses structured output and an evidence-ID protocol.

The model receives:

- the job requirements;
- deterministic match scores;
- matched/missing requirements;
- an identity-free evidence catalog such as `E001`, `E002`, ...

It does **not** receive candidate name, email, phone or location.

Every LLM-generated strength must reference one or more valid evidence IDs. HireFlow then resolves the evidence text server-side. Therefore the model cannot manufacture a quotation and pass it through as evidence.

The LLM is also forbidden from introducing new gaps: `gap_requirements` must be selected from the deterministic missing-requirement set.

## Offline fallback

`HeuristicGroundedEvaluator` produces a deterministic evidence-backed explanation when:

- no Gemini API key is configured;
- the Google SDK is not installed;
- a hosted-model request fails;
- a CI environment is offline.

This keeps the core system testable without making network access a prerequisite.

## Observed reranking on the supplied dataset

The TF-IDF retrieval baseline originally placed the Tax Manager candidate at rank 1. After requirement-aware reranking, the top 10 are:

| Rerank | Candidate | Role | Experience | Retrieval rank | Hybrid score |
|---:|---|---|---:|---:|---:|
| 1 | resume_35 | Accountant | 3 | 8 | 78.02 |
| 2 | resume_11 | Senior Accountant | 5 | 6 | 76.11 |
| 3 | resume_22 | Staff Accountant | 3 | 7 | 74.04 |
| 4 | resume_03 | Staff Accountant | 4 | 9 | 73.97 |
| 5 | resume_05 | Staff Accountant | 3 | 17 | 73.77 |
| 6 | resume_17 | Tax Manager | 5 | 1 | 73.16 |
| 7 | resume_02 | Senior Accountant | 7 | 10 | 72.87 |
| 8 | resume_49 | Staff Accountant | 2 | 18 | 70.24 |
| 9 | resume_32 | Staff Accountant | 3 | 13 | 70.02 |
| 10 | resume_43 | Accountant | 4 | 19 | 69.22 |

This is a qualitative sanity check, **not a performance claim**. The important observation is that role/experience/requirement signals can correct obvious lexical-retrieval ordering issues. Phase 4 will introduce ground-truth labels before reporting retrieval/ranking metrics.

## Run Phase 3 locally

After Phase 2 ingestion/indexing:

```bash
python scripts/rank_candidates.py \
  --index-dir artifacts/indexes/baseline_full_tfidf \
  --evaluator heuristic \
  --output artifacts/phase3_results.json
```

With Gemini configured in `.env`:

```bash
python scripts/rank_candidates.py \
  --index-dir artifacts/indexes/baseline_full_tfidf \
  --evaluator gemini
```

The Gemini path keeps deterministic ranking unchanged and only replaces the narrative explanation layer.
