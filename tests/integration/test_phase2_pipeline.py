from pathlib import Path

from hireflow.embeddings import TfidfEmbeddingModel
from hireflow.models import CandidateProfile, JobRequirements
from hireflow.retrieval import aggregate_candidate_hits, build_vector_index, candidate_documents


def test_tfidf_to_vector_search_pipeline() -> None:
    candidates = [
        CandidateProfile(
            candidate_id="accountant",
            source_file=Path("a.pdf"),
            headline="Senior Accountant",
            skills=["GAAP", "Financial Reporting", "Tax Preparation"],
            software=["Microsoft Excel", "QuickBooks"],
        ),
        CandidateProfile(
            candidate_id="engineer",
            source_file=Path("b.pdf"),
            headline="Software Engineer",
            skills=["Python", "Kubernetes", "APIs"],
        ),
    ]
    job = JobRequirements(
        job_id="j1",
        title="Senior Accountant",
        required_skills=["GAAP", "Financial Reporting", "Tax Preparation"],
        required_software=["Microsoft Excel"],
    )
    docs = [d for c in candidates for d in candidate_documents(c)]
    model = TfidfEmbeddingModel().fit([d.text for d in docs])
    vectors = model.embed_documents([d.text for d in docs])
    index = build_vector_index(vectors, docs, backend="numpy")
    hits = index.search(model.embed_query(job.matching_text()), k=2)
    results = aggregate_candidate_hits(hits, top_k_candidates=2)
    assert results[0].candidate_id == "accountant"
