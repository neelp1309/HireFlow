# Phase 2 Complete — Core ML Pipeline

Implemented and validated:

- PDF ingestion with typed failures
- text normalization and synthetic-placeholder removal
- section-aware resume parsing
- structured `CandidateProfile` generation
- structured Senior Accountant `JobRequirements` parsing
- privacy-aware retrieval text (identity excluded)
- full-resume and semantic-section retrieval documents
- reproducible TF-IDF embedding baseline
- Gemini embedding adapter for later dense retrieval experiments
- FAISS `IndexFlatIP` vector backend
- exact NumPy cosine fallback for offline/CI validation
- persisted vector/index metadata
- candidate-level aggregation from section hits
- CLI scripts for ingestion, indexing and search
- Phase 2 unit/integration smoke tests
- actual 50-resume dataset validation

Validation results for the supplied data:

- 50/50 resumes parsed
- 0 parse failures
- placeholder leakage after cleaning: 0
- 50/50 profiles contain skills, software and work experience
- baseline retrieval executed end to end
- full test suite: 12 passing tests

See `docs/phase2_core_ml_pipeline.md` for architecture and observed baseline behavior.
