# Phase 4 — Evaluation Layer

Phase 4 turns HireFlow from a system that merely "looks reasonable" into a system with measurable offline ranking behavior.

## What is evaluated

Four offline experiments are run on the same 50-candidate corpus:

1. TF-IDF full-resume semantic retrieval
2. TF-IDF section-aware semantic retrieval
3. TF-IDF full-resume retrieval + hybrid reranking
4. TF-IDF section-aware retrieval + hybrid reranking

A fixed-seed random ranking baseline is also averaged over 1,000 shuffles to provide context.

## Metrics

- **Precision@K** — fraction of top-K candidates with label >= 2.
- **Recall@K** — fraction of all benchmark-relevant candidates recovered in the top K.
- **Strong@K** — fraction of top-K candidates labeled 3 (strong fit).
- **MRR** — reciprocal rank of the first relevant candidate.
- **Average Precision** — ranking-wide binary relevance quality.
- **NDCG@K** — primary graded-relevance metric; rewards putting strong fits above moderate, weak, and unsuitable candidates.

Because the dataset has only one JD, **NDCG@10 is the primary descriptive metric**, not a claim of generalization.

## Experimental hygiene

Evaluation retrieves/reranks all 50 candidates, rather than only reranking the production Top-20 shortlist. This prevents the evaluation from hiding candidates that the first-stage retriever missed.

The hybrid weights remain the Phase 3 baseline. They are not optimized against Phase 4 labels because there is no separate validation job/query set.

## Dense embedding experiment

The experiment runner supports Gemini embedding indexes, but no dense-embedding result is reported unless a Gemini API key is configured and a corresponding index is built. This prevents fabricated or unverified benchmark claims.
