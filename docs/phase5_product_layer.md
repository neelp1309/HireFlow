# Phase 5 — Recruiter Product Layer

Phase 5 turns the evaluated HireFlow ML pipeline into an interactive decision-support product without moving model logic into the UI.

## Product architecture

`app/streamlit_app.py` is intentionally a presentation layer. End-to-end orchestration lives in `hireflow.product.HireFlowProductService`, which means a future REST API, batch worker, or desktop client can reuse the same pipeline.

```text
Streamlit
   │
   ├── PDF / ZIP upload
   ├── run configuration
   └── recruiter filters
   │
   ▼
HireFlowProductService
   ├── parsing
   ├── embeddings
   ├── vector search
   ├── candidate aggregation
   ├── hybrid reranking
   └── grounded evaluation
   │
   ▼
ProductSearchBundle
   │
   ├── ranked shortlist
   ├── candidate details
   ├── evidence
   └── export helpers
```

## Recruiter workflow

The application accepts one JD PDF plus either multiple resume PDFs or a ZIP of resume PDFs. Interactive uploads are written only to a temporary workspace for parsing and are not copied to repository data folders.

The shortlist view provides:

- deterministic overall fit score;
- recommendation tier;
- experience and current role;
- matched requirements and resume-evidence gaps;
- filtering by fit score, recommendation, role text, and experience;
- CSV and JSON export.

The candidate-detail view provides the five score components, grounded reasoning, resume evidence, education/software information, and an explicitly separated recruiter contact panel.

## Responsible-use design

Candidate name, email, phone, location, and LinkedIn are useful to a recruiter but are intentionally separated from scoring. They are excluded from matching text, the Phase 4 benchmark, deterministic evidence matching, and Gemini prompts.

HireFlow reports an absent requirement as "direct evidence not found in the resume" rather than claiming the candidate lacks the skill. The UI also labels the system as decision support rather than an autonomous hiring decision maker.

## Offline and Gemini modes

The default product mode uses TF-IDF embeddings and the grounded heuristic evaluator, so the complete application can run without external AI services. If `GEMINI_API_KEY` is configured, recruiters can opt into Gemini dense embeddings and/or the grounded Gemini explanation layer.

Gemini does not control the numeric fit score.

## Validation

Phase 5 adds product-service unit/integration tests and an offline smoke script. The product service was also exercised against all 50 supplied resumes and the supplied Senior Accountant JD. The validated default shortlist was:

1. `resume_35` — Accountant — 3 years — 78.02
2. `resume_11` — Senior Accountant — 5 years — 76.11
3. `resume_22` — Staff Accountant — 3 years — 74.04
4. `resume_03` — Staff Accountant — 4 years — 73.97
5. `resume_05` — Staff Accountant — 3 years — 73.77

This matches the Phase 4 hybrid-ranking behavior.

The current execution environment does not ship Streamlit and has no internet package access, so the UI module was syntax-compiled while its orchestration layer was executed and tested directly. `streamlit>=1.40,<2` remains declared in `requirements.txt` for local execution.
