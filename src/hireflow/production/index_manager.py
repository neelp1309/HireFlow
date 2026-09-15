"""Persistent corpus index synchronization for production-style reuse.

Dense embedding spaces are fixed, so unchanged document vectors can be reused when
the corpus changes. TF-IDF depends on the full corpus vocabulary; any change therefore
triggers a full refit, while an unchanged corpus loads the persisted model/index.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from hireflow.config import Settings, get_settings
from hireflow.embeddings import GeminiEmbeddingModel, TfidfEmbeddingModel
from hireflow.exceptions import ConfigurationError, RetrievalError
from hireflow.models import CandidateProfile, RetrievalDocument
from hireflow.models.schemas import StrictModel
from hireflow.retrieval import build_vector_index, candidate_documents, load_vector_index


class IndexSyncReport(StrictModel):
    index_name: str
    embedding_provider: str
    document_mode: str
    vector_backend: str
    total_documents: int = Field(ge=0)
    embedded_documents: int = Field(ge=0)
    reused_embeddings: int = Field(ge=0)
    removed_documents: int = Field(ge=0)
    full_rebuild: bool
    corpus_changed: bool
    elapsed_ms: float = Field(ge=0)


class SyncedCorpusIndex:
    def __init__(self, embedding_model, vector_index, documents: list[RetrievalDocument], report: IndexSyncReport) -> None:
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.documents = documents
        self.report = report


def _doc_hash(document: RetrievalDocument) -> str:
    payload = json.dumps(document.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CorpusIndexManager:
    """Synchronize a reusable candidate corpus into a private artifact directory."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _embedding_model(self, provider: str):
        if provider == "gemini":
            if not self.settings.gemini_api_key:
                raise ConfigurationError("GEMINI_API_KEY is required for dense incremental indexing")
            return GeminiEmbeddingModel(
                self.settings.gemini_api_key,
                self.settings.gemini_embedding_model,
                max_retries=self.settings.gemini_max_retries,
                initial_backoff_seconds=self.settings.gemini_initial_backoff_seconds,
            )
        return TfidfEmbeddingModel()

    def _directory(self, index_name: str) -> Path:
        safe = "".join(ch for ch in index_name if ch.isalnum() or ch in {"-", "_"}).strip("-_")
        if not safe:
            raise ValueError("index_name must contain at least one alphanumeric character")
        return self.settings.vector_index_dir / safe

    def sync(
        self,
        index_name: str,
        candidates: list[CandidateProfile],
        *,
        embedding_provider: Literal["tfidf", "gemini"] = "tfidf",
        document_mode: Literal["full", "section"] = "full",
        vector_backend: Literal["auto", "faiss", "numpy"] = "auto",
    ) -> SyncedCorpusIndex:
        started = time.perf_counter()
        if not candidates:
            raise ValueError("At least one candidate is required to build an index")
        documents = [
            document
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
            for document in candidate_documents(candidate, mode=document_mode)
        ]
        hashes = {document.vector_id: _doc_hash(document) for document in documents}
        target = self._directory(index_name)
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        vectors_path = target / "embeddings.npy"
        model_path = target / "embedding_model.joblib"
        index_path = target / "vector_index"

        old_manifest: dict = {}
        if manifest_path.exists():
            try:
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RetrievalError(f"Corrupt index manifest: {manifest_path}") from exc

        compatible = (
            old_manifest.get("embedding_provider") == embedding_provider
            and old_manifest.get("document_mode") == document_mode
            and old_manifest.get("embedding_model")
            == (self.settings.gemini_embedding_model if embedding_provider == "gemini" else "tfidf-word-1-2")
        )
        old_hashes: dict[str, str] = old_manifest.get("document_hashes", {}) if compatible else {}
        changed_ids = {vector_id for vector_id, digest in hashes.items() if old_hashes.get(vector_id) != digest}
        removed_ids = set(old_hashes) - set(hashes)
        corpus_changed = bool(changed_ids or removed_ids or not compatible)

        # Exact cache hit: load persisted artifacts, performing no embedding work.
        if not corpus_changed and index_path.exists():
            if embedding_provider == "tfidf":
                model = TfidfEmbeddingModel.load(model_path)
            else:
                model = self._embedding_model("gemini")
            index = load_vector_index(index_path)
            report = IndexSyncReport(
                index_name=index_name,
                embedding_provider=embedding_provider,
                document_mode=document_mode,
                vector_backend=index.backend_name,
                total_documents=len(documents),
                embedded_documents=0,
                reused_embeddings=len(documents),
                removed_documents=0,
                full_rebuild=False,
                corpus_changed=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return SyncedCorpusIndex(model, index, documents, report)

        model = self._embedding_model(embedding_provider)
        full_rebuild = embedding_provider == "tfidf" or not compatible or not vectors_path.exists()
        if full_rebuild:
            model.fit([document.text for document in documents])
            vectors = model.embed_documents([document.text for document in documents])
            embedded_documents = len(documents)
            reused_embeddings = 0
        else:
            # Gemini embeddings are corpus-independent: reuse vectors for unchanged docs.
            old_vectors = np.load(vectors_path)
            old_vector_ids: list[str] = old_manifest.get("vector_ids", [])
            if len(old_vector_ids) != len(old_vectors):
                raise RetrievalError("Persisted vector IDs do not match embeddings matrix")
            old_by_id = {vector_id: old_vectors[idx] for idx, vector_id in enumerate(old_vector_ids)}
            new_docs = [document for document in documents if document.vector_id in changed_ids]
            fresh = model.embed_documents([document.text for document in new_docs]) if new_docs else np.empty((0, 0))
            fresh_by_id = {document.vector_id: fresh[idx] for idx, document in enumerate(new_docs)}
            combined = []
            for document in documents:
                if document.vector_id in fresh_by_id:
                    combined.append(fresh_by_id[document.vector_id])
                else:
                    combined.append(old_by_id[document.vector_id])
            vectors = np.vstack(combined).astype(np.float32)
            embedded_documents = len(new_docs)
            reused_embeddings = len(documents) - embedded_documents

        index = build_vector_index(vectors, documents, backend=vector_backend)
        np.save(vectors_path, vectors)
        index.save(index_path)
        if embedding_provider == "tfidf":
            model.save(model_path)
        manifest = {
            "schema_version": 1,
            "embedding_provider": embedding_provider,
            "embedding_model": model.model_name,
            "document_mode": document_mode,
            "vector_backend": index.backend_name,
            "vector_ids": [document.vector_id for document in documents],
            "document_hashes": hashes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        report = IndexSyncReport(
            index_name=index_name,
            embedding_provider=embedding_provider,
            document_mode=document_mode,
            vector_backend=index.backend_name,
            total_documents=len(documents),
            embedded_documents=embedded_documents,
            reused_embeddings=reused_embeddings,
            removed_documents=len(removed_ids),
            full_rebuild=full_rebuild,
            corpus_changed=True,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return SyncedCorpusIndex(model, index, documents, report)


__all__ = ["CorpusIndexManager", "IndexSyncReport", "SyncedCorpusIndex"]
