"""Local TF-IDF embedding baseline for reproducible offline experiments."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from hireflow.exceptions import EmbeddingError


class TfidfEmbeddingModel:
    """A deterministic lexical embedding baseline.

    This is deliberately retained even when dense Gemini embeddings are added:
    it gives us a cheap, reproducible baseline to beat in later experiments.
    """

    def __init__(self, max_features: int = 4096, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "tfidf-word-1-2"

    def fit(self, documents: list[str]) -> "TfidfEmbeddingModel":
        if not documents:
            raise EmbeddingError("Cannot fit TF-IDF on an empty document collection")
        self.vectorizer.fit(documents)
        self._fitted = True
        return self

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise EmbeddingError("TF-IDF embedding model has not been fitted")

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        self._ensure_fitted()
        matrix = self.vectorizer.transform(documents)
        matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.toarray().astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, target)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfEmbeddingModel":
        obj = cls()
        obj.vectorizer = joblib.load(Path(path))
        obj._fitted = True
        return obj
