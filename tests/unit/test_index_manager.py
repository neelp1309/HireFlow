from pathlib import Path

from hireflow.config import Settings
from hireflow.models import CandidateProfile
from hireflow.production import CorpusIndexManager


def profile(candidate_id: str, summary: str) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        source_file=Path(f"{candidate_id}.pdf"),
        headline="Accountant",
        professional_summary=summary,
    )


def test_tfidf_corpus_index_reuses_unchanged_snapshot(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        vector_index_dir=tmp_path / "indexes",
        artifacts_dir=tmp_path / "artifacts",
    )
    manager = CorpusIndexManager(settings)
    candidates = [profile("c1", "financial reporting"), profile("c2", "tax accounting")]
    first = manager.sync("test", candidates, vector_backend="numpy")
    second = manager.sync("test", candidates, vector_backend="numpy")
    assert first.report.full_rebuild is True
    assert first.report.embedded_documents == 2
    assert second.report.corpus_changed is False
    assert second.report.embedded_documents == 0
    assert second.report.reused_embeddings == 2


def test_tfidf_change_requires_refit(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        vector_index_dir=tmp_path / "indexes",
        artifacts_dir=tmp_path / "artifacts",
    )
    manager = CorpusIndexManager(settings)
    manager.sync("test", [profile("c1", "financial reporting")], vector_backend="numpy")
    changed = manager.sync("test", [profile("c1", "financial reporting and tax")], vector_backend="numpy")
    assert changed.report.corpus_changed is True
    assert changed.report.full_rebuild is True
    assert changed.report.embedded_documents == 1


def test_dense_sync_reuses_unchanged_embeddings(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    class FakeDenseModel:
        model_name = "gemini-embedding-001"

        def __init__(self) -> None:
            self.embedded_batches: list[list[str]] = []

        def fit(self, documents: list[str]):
            return self

        def embed_documents(self, documents: list[str]) -> np.ndarray:
            self.embedded_batches.append(list(documents))
            rows = []
            for text in documents:
                value = float((sum(text.encode("utf-8")) % 97) + 1)
                rows.append([value, value / 2.0, 1.0])
            return np.asarray(rows, dtype=np.float32)

        def embed_query(self, text: str) -> np.ndarray:
            return self.embed_documents([text])[0]

    settings = Settings(
        _env_file=None,
        app_env="test",
        gemini_api_key="fake-key",
        vector_index_dir=tmp_path / "indexes",
        artifacts_dir=tmp_path / "artifacts",
    )
    manager = CorpusIndexManager(settings)
    created: list[FakeDenseModel] = []

    def fake_model(_provider: str):
        model = FakeDenseModel()
        created.append(model)
        return model

    monkeypatch.setattr(manager, "_embedding_model", fake_model)
    initial_candidates = [profile("c1", "financial reporting"), profile("c2", "tax accounting")]
    first = manager.sync(
        "dense",
        initial_candidates,
        embedding_provider="gemini",
        vector_backend="numpy",
    )
    assert first.report.embedded_documents == 2

    updated_candidates = [profile("c1", "financial reporting"), profile("c2", "tax and audit accounting")]
    second = manager.sync(
        "dense",
        updated_candidates,
        embedding_provider="gemini",
        vector_backend="numpy",
    )
    assert second.report.full_rebuild is False
    assert second.report.embedded_documents == 1
    assert second.report.reused_embeddings == 1
    assert len(created[-1].embedded_batches) == 1
    assert len(created[-1].embedded_batches[0]) == 1
