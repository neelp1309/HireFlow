# Interview Guide

## 60-second project explanation

HireFlow is an end-to-end candidate search and evaluation system. Resume PDFs are parsed into validated candidate profiles, identity fields are separated from qualification text, and candidates are retrieved using TF-IDF or optional dense embeddings. Retrieval results are aggregated at candidate level and then reranked with deterministic job-fit features such as required-skill coverage, experience, role/seniority and preferred criteria. I evaluate ranking quality with graded relevance labels and NDCG rather than only eyeballing results. Gemini is optional and is used only to generate evidence-backed explanations; it cannot modify the numeric fit score. The same service powers Streamlit and FastAPI, with caching, persistent indexing, observability, Docker and CI.

## Why not send all resumes directly to an LLM?

- cost and latency scale poorly;
- rankings become less reproducible;
- long-context limits become a bottleneck;
- numeric scores become hard to audit;
- retrieval metrics become impossible to isolate.

HireFlow narrows the corpus first, reranks deterministically, then uses an LLM only on a small shortlist.

## Why keep TF-IDF if Gemini embeddings exist?

A baseline is necessary to know whether added complexity actually improves the system. TF-IDF is cheap, deterministic and easy to debug. The project preserves it as an offline baseline and makes dense embeddings optional.

## Why did section-aware TF-IDF perform worse?

The supplied resumes are template-heavy. Splitting sections removed some cross-section context, while individual sections contained generic accounting vocabulary shared by many candidates. That reduced ranking discrimination. The experiment is kept because negative results are useful engineering evidence.

## Why use NDCG?

The benchmark is graded (`strong`, `moderate`, `weak`, `not suitable`). NDCG rewards putting stronger candidates earlier, whereas binary Precision@K treats all relevant candidates equally.

## How does hybrid ranking work?

The baseline score uses configurable weighted components:

- semantic relevance — 25%
- required requirements — 30%
- experience — 15%
- role/seniority — 20%
- preferred criteria — 10%

The weights are hypotheses, not universal business truth. They were not tuned against the one-JD benchmark to avoid overfitting.

## How do you control LLM hallucination?

1. The LLM does not produce the numeric score.
2. The prompt uses qualification evidence rather than raw identity/contact data.
3. Evidence is assigned server-side IDs.
4. The model can reference only those IDs.
5. Unknown evidence IDs fail validation.
6. Missing evidence is worded as “not found in the resume,” not as proof that a candidate lacks a skill.

## How would you scale from 50 to 1,000,000 resumes?

- use an external vector database or distributed ANN index;
- precompute dense resume embeddings offline;
- store structured metadata separately;
- add metadata filtering before ANN search;
- use asynchronous ingestion workers;
- cache repeated job queries;
- use a learned/cross-encoder reranker only on a small retrieved set;
- put the API behind autoscaling and external distributed caching;
- add tenant isolation, encryption and authorization.

## Why can TF-IDF not reuse only one changed document embedding?

TF-IDF coordinates depend on corpus-wide document frequencies. Adding/removing a document can change IDF values for every document, so the model must be refit when the corpus changes. Dense embeddings are corpus-independent and can reuse unchanged document vectors.

## What would you improve with more time/data?

- multi-JD human-reviewed benchmark;
- train/validation/test split by job query;
- dense embedding comparison;
- cross-encoder reranking;
- explicit calibration of fit scores;
- fairness and subgroup analysis;
- extraction quality evaluation;
- ATS integrations;
- persistent encrypted candidate store;
- distributed tracing and external cache/vector store.

## Honest limitations to mention

- one evaluated JD;
- 50 synthetic/template-heavy resumes;
- silver labels rather than independent expert annotations;
- accounting-focused matching taxonomy;
- live Gemini performance not included in the benchmark results.

These limitations are strengths in an interview if you explain how you would design the next experiment rather than pretending the system is production-validated.
