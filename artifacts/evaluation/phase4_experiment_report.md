# Phase 4 Experiment Report

## Benchmark

- Candidates: **50**
- Binary relevance threshold: **label >= 2**
- Labels: 0=not suitable, 1=weak, 2=moderate, 3=strong
- Status: **project-authored silver benchmark; independent human review still required**

## Ranking results

| Experiment | P@5 | P@10 | Strong@5 | Strong@10 | NDCG@5 | NDCG@10 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF full-resume retrieval | 0.600 | 0.800 | 0.000 | 0.400 | 0.338 | 0.534 | 0.759 |
| TF-IDF section-aware retrieval | 0.600 | 0.700 | 0.000 | 0.200 | 0.318 | 0.411 | 0.699 |
| TF-IDF full-resume + hybrid reranking | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 | 0.926 | 0.988 |
| TF-IDF section-aware + hybrid reranking | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 | 0.918 | 0.989 |
| Random baseline (mean) | 0.659 | 0.659 | 0.263 | 0.264 | 0.461 | 0.461 | 0.685 |

## Interpretation constraints

- There is only **one supplied job description**, so these metrics are descriptive for this JD and are not a generalization estimate.
- The benchmark is a **silver benchmark**, authored from an explicit rubric and not external human ground truth.
- Hybrid weights were **not optimized against these labels**. Tuning on this single query would overfit the benchmark.
- Gemini dense-embedding experiments are supported by the codebase but are not executed without a configured API key.
