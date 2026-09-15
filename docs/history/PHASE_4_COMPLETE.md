# Phase 4 Complete — Evaluation Layer

Phase 4 adds a reproducible offline benchmark and ranking-evaluation framework to HireFlow.

## Delivered

- PII-free 50-candidate silver relevance benchmark for the supplied Senior Accountant JD
- explicit 0-3 relevance rubric and annotation documentation
- Precision@K, Recall@K, StrongFit@K, MRR, Average Precision and NDCG@K
- full-corpus evaluation (all 50 candidates) to avoid Top-K censoring during experiments
- four reproducible experiments:
  - TF-IDF full-resume retrieval
  - TF-IDF section-aware retrieval
  - full-resume retrieval + hybrid reranking
  - section-aware retrieval + hybrid reranking
- 1,000-shuffle deterministic random baseline
- JSON, CSV and Markdown experiment artifacts
- candidate-level ranking-comparison CSV
- benchmark validation and metric unit tests
- Phase 4 smoke validation

## Dataset-specific result

On the project-authored silver benchmark, the full-resume TF-IDF baseline achieved NDCG@10 **0.534**, while full-resume + hybrid reranking achieved **0.926**. The section-aware hybrid variant achieved **0.918**.

Top-5 strong-fit rate changed from **0.00** for both lexical retrieval baselines to **1.00** after hybrid reranking.

These are **descriptive results for the single supplied JD**, not a claim of generalization or production accuracy. The labels are a silver benchmark and still require independent human review for production-grade conclusions. Hybrid weights were deliberately not optimized against this benchmark.

## Validation

- 30 automated tests passing
- Phase 1 validation passed
- Phase 2 validation passed
- Phase 3 validation passed
- Phase 4 validation passed
