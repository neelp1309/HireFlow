# Demo Guide

This guide is designed for a GitHub reviewer or interviewer who wants to understand HireFlow quickly.

## 1. Fastest demo — public synthetic data

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
make demo
```

The public demo uses three synthetic qualification-only candidate profiles and does not require the original resume PDFs.

Expected ordering:

1. `demo_candidate_01` — Senior Accountant, 4 YOE
2. a less-aligned adjacent-role candidate
3. a junior accounting candidate

The script exits non-zero if candidate 01 does not rank first.

## 2. Streamlit product demo

```bash
make run
```

Then upload:

- one job-description PDF;
- multiple resume PDFs, or a ZIP containing PDFs.

Recommended walkthrough:

1. Run the default TF-IDF + heuristic configuration.
2. Show the shortlist.
3. Open a candidate detail view.
4. Explain the five score components.
5. Expand evidence used for strengths/gaps.
6. Show that contact information is separated from scoring.
7. Export CSV/JSON.
8. Open the Model & Evaluation page.

## 3. API demo

```bash
make api
```

Open:

```text
http://localhost:8000/docs
```

Useful calls:

```text
GET /health/live
GET /health/ready
GET /metrics
POST /v1/search/structured
POST /v1/search/files
```

## 4. Evaluation demo

With the private supplied dataset available locally:

```bash
make evaluate
```

Then inspect:

```text
artifacts/evaluation/phase4_metrics.csv
artifacts/evaluation/phase4_experiment_report.md
artifacts/evaluation/phase4_ranking_comparison.csv
```

## 5. What to emphasize during a live interview

Do not position HireFlow as “I used Gemini to rank resumes.” The stronger story is:

> I built a two-stage candidate retrieval and ranking system, retained a simple retrieval baseline, measured ranking quality with graded relevance metrics, added explicit requirement-aware reranking, constrained the LLM to grounded explanation only, and then productionized the pipeline behind Streamlit and FastAPI with caching, observability, testing, Docker and CI.
