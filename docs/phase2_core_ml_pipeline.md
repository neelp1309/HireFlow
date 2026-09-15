# Phase 2 — Core ML Pipeline

Phase 2 turns the Phase 1 contracts into a working retrieval pipeline over the supplied
Senior Accountant dataset.

## Pipeline

```text
Resume PDFs (50)
      │
      ▼
PyPDF text extraction
      │
      ▼
Deterministic cleaning
  - normalize bullets
  - remove synthetic placeholders
  - collapse PDF line noise
      │
      ▼
Section-aware parsing
  - summary
  - skills
  - software
  - experience
  - education
  - certifications
      │
      ▼
CandidateProfile JSONL
      │
      ▼
Qualification-only retrieval text
      │
      ▼
Embedding model
  ├─ TF-IDF baseline (offline/reproducible)
  └─ Gemini dense adapter (API-ready)
      │
      ▼
Vector index
  ├─ FAISS IndexFlatIP (preferred runtime)
  └─ NumPy exact cosine fallback (CI/offline)
      │
      ▼
Top-K vector hits
      │
      ▼
Candidate-level aggregation
      │
      ▼
Baseline ranking
```

The Job Description follows the same ingestion/cleaning principle but is parsed into
`JobRequirements`. Benefits, contact details and company marketing language are excluded
from `matching_text()`.

## Why keep TF-IDF when the target architecture uses dense embeddings?

TF-IDF gives the project a deterministic baseline. Later dense/hybrid/reranking experiments
must beat a known baseline instead of being accepted because they "look semantic". This is
important for ML engineering evaluation and ablation studies.

## Why FAISS plus a NumPy fallback?

FAISS is the intended local vector backend and the repository contains an IndexFlatIP
implementation. A NumPy exact cosine backend implements the same interface so CI and offline
validation remain possible on environments where the native `faiss-cpu` wheel is unavailable.
For this 50-resume dataset both are exact-search backends; at scale FAISS is the production
path.

## Actual supplied-dataset validation

The Phase 2 pipeline was exercised against all 50 supplied resume PDFs and the supplied
Senior Accountant JD.

- resume PDFs discovered: **50**
- resumes parsed successfully: **50**
- parse failures: **0**
- resumes with extracted skills: **50/50**
- resumes with extracted software: **50/50**
- resumes with extracted work experience: **50/50**
- synthetic `{generate_achievements(...)}` placeholder remaining after cleaning: **0**
- parsed JD title: **Senior Accountant**
- parsed JD experience range: **3–5 years**

The baseline full-resume TF-IDF index contained 50 vectors with 1,296 dimensions in this
run. The section-mode index contained 200 vectors (four semantic section groups per
candidate where present).

## Baseline result is intentionally not the final ranking

The lexical baseline surfaces accounting-related candidates but does not reason about
seniority or required-vs-preferred criteria. For example, a Tax Manager can outrank a
Senior Accountant because it shares many tax/accounting terms. This is a useful baseline
failure, not something to hide: Phase 3 will add explicit experience/role/skill matching,
hybrid scoring and reranking to address it.

Top full-resume baseline results from the supplied dataset (candidate IDs only):

| Rank | Candidate ID | Headline | Experience | TF-IDF score |
|---:|---|---|---:|---:|
| 1 | resume_17 | Tax Manager | 5 | 0.2828 |
| 2 | resume_14 | Accountant | 2 | 0.2681 |
| 3 | resume_50 | Accounting Assistant | 1 | 0.2512 |
| 4 | resume_13 | Financial Analyst | 2 | 0.2450 |
| 5 | resume_26 | Senior Financial Analyst | 6 | 0.2427 |
| 6 | resume_11 | Senior Accountant | 5 | 0.2361 |
| 7 | resume_22 | Staff Accountant | 3 | 0.2322 |
| 8 | resume_35 | Accountant | 3 | 0.2317 |
| 9 | resume_03 | Staff Accountant | 4 | 0.2293 |
| 10 | resume_02 | Senior Accountant | 7 | 0.2276 |

No claim is made yet that this ordering is correct. Phase 4 will create a labeled relevance
benchmark before retrieval metrics such as Recall@K/NDCG are reported.

## CLI workflow

Place private input files locally, then run:

```bash
python scripts/ingest_dataset.py \
  --resumes-dir data/raw/resumes \
  --job-pdf data/raw/jobs/Senior_Accountant_Position.pdf

python scripts/build_index.py \
  --document-mode full \
  --embedding tfidf \
  --vector-backend auto

python scripts/search_candidates.py --top-k 10
```

To prepare the section-aware experiment:

```bash
python scripts/build_index.py \
  --output-dir artifacts/indexes/section_tfidf \
  --document-mode section \
  --embedding tfidf
```

For Gemini embeddings, set `GEMINI_API_KEY` and use `--embedding gemini`. This adapter is
implemented but API execution requires a user-provided key and network access.
