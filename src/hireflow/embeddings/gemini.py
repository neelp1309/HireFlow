"""Gemini dense embeddings adapter with bounded exponential-backoff retries."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hireflow.exceptions import EmbeddingError


class GeminiEmbeddingModel:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        *,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise EmbeddingError("Gemini API key is required for Gemini embeddings")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.api_key = api_key
        self._model = model
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise EmbeddingError(
                "google-genai is not installed. Install project requirements to use Gemini embeddings."
            ) from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def fit(self, documents: list[str]) -> "GeminiEmbeddingModel":
        return self

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        if not texts:
            raise EmbeddingError("Cannot embed an empty text collection")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._get_client().models.embed_content(
                    model=self._model,
                    contents=texts,
                    config={"task_type": task_type},
                )
                vectors = [np.asarray(item.values, dtype=np.float32) for item in response.embeddings]
                matrix = np.vstack(vectors)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return matrix / norms
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.initial_backoff_seconds * (2 ** (attempt - 1)))
        raise EmbeddingError(
            f"Gemini embedding request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        return self._embed(documents, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._model, encoding="utf-8")
