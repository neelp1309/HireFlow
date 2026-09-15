# Evaluation Results

## Scope

The current benchmark evaluates one supplied **Senior Accountant** job description against 50 supplied accounting/finance resumes. The labels are project-authored silver labels using an explicit rubric and are therefore suitable for architecture comparison, not for claiming general real-world hiring accuracy.

## Label distribution

- 3 — Strong fit: 13 candidates
- 2 — Moderate fit: 20 candidates
- 1 — Weak fit: 10 candidates
- 0 — Not suitable: 7 candidates

A relevance threshold of `>= 2` is used for binary Precision/Recall-style metrics. NDCG uses the graded labels directly.

## Experiment summary

| Experiment | P@5 | P@10 | R@5 | R@10 | Strong@5 | Strong@10 | NDCG@5 | NDCG@10 | MRR | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF full-resume retrieval | 0.60 | 0.80 | 0.091 | 0.242 | 0.00 | 0.40 | 0.338 | 0.534 | 1.00 | 0.759 |
| TF-IDF section-aware retrieval | 0.60 | 0.70 | 0.091 | 0.212 | 0.00 | 0.20 | 0.318 | 0.411 | 1.00 | 0.699 |
| **Full-resume + hybrid reranking** | **1.00** | **1.00** | **0.152** | **0.303** | **1.00** | **0.80** | **1.000** | **0.926** | **1.00** | 0.988 |
| Section-aware + hybrid reranking | 1.00 | 1.00 | 0.152 | 0.303 | 1.00 | 0.80 | 1.000 | 0.918 | 1.00 | **0.989** |

## Interpretation

The strongest improvement came from requirement-aware hybrid reranking rather than from changing the TF-IDF document granularity.

The full-resume lexical baseline retrieved accounting-related resumes but often over-valued candidates from adjacent roles because it could not distinguish semantic similarity from job fit. The hybrid reranker explicitly considers role/seniority, experience range, required accounting concepts, software/education, and preferred qualifications.

Section-aware TF-IDF did not improve this dataset. The supplied resumes are highly templated, and isolated sections often contain generic accounting language without enough role/experience context. Preserving this negative result is intentional.

## Why NDCG is the primary metric

Candidate relevance is graded. A top-five list containing five moderate candidates is not equivalent to one containing five strong candidates even though binary Precision@5 could be identical. NDCG rewards ordering stronger candidates earlier.

## Runtime smoke measurements

Using TF-IDF + NumPy exact search + heuristic grounded evaluation on the supplied 50-resume dataset in the build environment:

| Measurement | Result |
|---|---:|
| Cold file-based run | ~576 ms |
| Repeated request | ~299 ms |
| Warm post-parsing cache lookup | ~1.3 ms |
| First persistent TF-IDF sync | 50 embedded / 0 reused |
| Unchanged second sync | 0 embedded / 50 reused |

These are environment-specific smoke measurements and should not be interpreted as production SLAs.

## Evaluation limitations

1. Only one job description is labeled.
2. The resumes are synthetic/template-heavy.
3. Labels were authored for the project and have not been independently adjudicated.
4. Hybrid weights were deliberately not tuned against this one benchmark because doing so would overfit the evaluation set.
5. No fairness/generalization claim is made.
6. Dense Gemini embedding results are not reported without an actual configured API experiment.

## Next evaluation milestone

A production-quality study should add multiple roles/JDs, independent human relevance judgments, train/validation/test query splits, inter-annotator agreement, dense-retrieval baselines, and fairness/subgroup evaluation where legally and ethically appropriate.
