import numpy as np

from hireflow.models import CandidateProfile, RetrievalDocument, RetrievalHit
from hireflow.retrieval import NumpyVectorIndex, aggregate_candidate_hits, candidate_documents


def test_numpy_index_returns_highest_cosine_match() -> None:
    docs = [
        RetrievalDocument(vector_id="a", candidate_id="c1", section="full", text="a"),
        RetrievalDocument(vector_id="b", candidate_id="c2", section="full", text="b"),
    ]
    index = NumpyVectorIndex.build(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), docs)
    hits = index.search(np.array([0.9, 0.1], dtype=np.float32), k=2)
    assert hits[0].candidate_id == "c1"
    assert hits[0].score > hits[1].score


def test_candidate_aggregation_groups_sections() -> None:
    hits = [
        RetrievalHit(vector_id="c1:s", candidate_id="c1", section="skills", score=0.9, text="x"),
        RetrievalHit(vector_id="c1:e", candidate_id="c1", section="experience", score=0.6, text="y"),
        RetrievalHit(vector_id="c2:s", candidate_id="c2", section="skills", score=0.7, text="z"),
    ]
    results = aggregate_candidate_hits(hits, top_k_candidates=2)
    assert results[0].candidate_id == "c1"
    assert len(results[0].hits) == 2


def test_candidate_documents_exclude_identity() -> None:
    candidate = CandidateProfile(
        candidate_id="c1",
        source_file="resume.pdf",
        full_name="Sensitive Name",
        headline="Senior Accountant",
        professional_summary="Financial reporting specialist",
        raw_text="Sensitive Name sensitive@example.com",
    )
    doc = candidate_documents(candidate, mode="full")[0]
    assert "Sensitive Name" not in doc.text
    assert "sensitive@example.com" not in doc.text
    assert "Senior Accountant" in doc.text
