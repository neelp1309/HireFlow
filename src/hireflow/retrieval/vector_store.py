"""Vector index implementations with FAISS-first and NumPy fallback support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from hireflow.exceptions import RetrievalError
from hireflow.models import RetrievalDocument, RetrievalHit


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class NumpyVectorIndex:
    """Exact cosine-similarity fallback used in CI/offline environments."""

    backend_name = "numpy-exact"

    def __init__(self, vectors: np.ndarray, documents: list[RetrievalDocument]) -> None:
        if len(vectors) != len(documents):
            raise RetrievalError("Vector/document count mismatch")
        self.vectors = _normalize_rows(vectors)
        self.documents = documents

    @classmethod
    def build(cls, vectors: np.ndarray, documents: list[RetrievalDocument]) -> "NumpyVectorIndex":
        return cls(vectors, documents)

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievalHit]:
        if k <= 0:
            return []
        q = _normalize_rows(query_vector)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[: min(k, len(scores))]
        return [
            RetrievalHit(
                vector_id=self.documents[i].vector_id,
                candidate_id=self.documents[i].candidate_id,
                section=self.documents[i].section,
                score=float(np.clip(scores[i], -1.0, 1.0)),
                text=self.documents[i].text,
                metadata=self.documents[i].metadata,
            )
            for i in top
        ]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "vectors.npy", self.vectors)
        (target / "documents.json").write_text(
            json.dumps([doc.model_dump(mode="json") for doc in self.documents], indent=2),
            encoding="utf-8",
        )
        (target / "backend.txt").write_text(self.backend_name, encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "NumpyVectorIndex":
        source = Path(directory)
        vectors = np.load(source / "vectors.npy")
        raw = json.loads((source / "documents.json").read_text(encoding="utf-8"))
        documents = [RetrievalDocument.model_validate(item) for item in raw]
        return cls(vectors, documents)


class FaissVectorIndex:
    """Exact inner-product FAISS index over L2-normalized vectors."""

    backend_name = "faiss-indexflatip"

    def __init__(self, index, documents: list[RetrievalDocument]) -> None:
        self.index = index
        self.documents = documents

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise RetrievalError(
                "faiss-cpu is not installed. Install project requirements or use backend='numpy'."
            ) from exc
        return faiss

    @classmethod
    def build(cls, vectors: np.ndarray, documents: list[RetrievalDocument]) -> "FaissVectorIndex":
        if len(vectors) != len(documents):
            raise RetrievalError("Vector/document count mismatch")
        faiss = cls._faiss()
        matrix = _normalize_rows(vectors)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index, documents)

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievalHit]:
        q = _normalize_rows(query_vector)
        scores, indices = self.index.search(q, min(k, len(self.documents)))
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            hits.append(
                RetrievalHit(
                    vector_id=doc.vector_id,
                    candidate_id=doc.candidate_id,
                    section=doc.section,
                    score=float(np.clip(score, -1.0, 1.0)),
                    text=doc.text,
                    metadata=doc.metadata,
                )
            )
        return hits

    def save(self, directory: str | Path) -> None:
        faiss = self._faiss()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "index.faiss"))
        (target / "documents.json").write_text(
            json.dumps([doc.model_dump(mode="json") for doc in self.documents], indent=2),
            encoding="utf-8",
        )
        (target / "backend.txt").write_text(self.backend_name, encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorIndex":
        faiss = cls._faiss()
        source = Path(directory)
        index = faiss.read_index(str(source / "index.faiss"))
        raw = json.loads((source / "documents.json").read_text(encoding="utf-8"))
        documents = [RetrievalDocument.model_validate(item) for item in raw]
        return cls(index, documents)


def build_vector_index(
    vectors: np.ndarray,
    documents: list[RetrievalDocument],
    backend: Literal["auto", "faiss", "numpy"] = "auto",
):
    if backend == "numpy":
        return NumpyVectorIndex.build(vectors, documents)
    if backend == "faiss":
        return FaissVectorIndex.build(vectors, documents)
    try:
        return FaissVectorIndex.build(vectors, documents)
    except RetrievalError:
        return NumpyVectorIndex.build(vectors, documents)


def load_vector_index(directory: str | Path):
    source = Path(directory)
    backend = (source / "backend.txt").read_text(encoding="utf-8").strip()
    if backend == FaissVectorIndex.backend_name:
        return FaissVectorIndex.load(source)
    if backend == NumpyVectorIndex.backend_name:
        return NumpyVectorIndex.load(source)
    raise RetrievalError(f"Unknown vector index backend: {backend}")
