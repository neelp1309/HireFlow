"""Embedding model contracts."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    @property
    def model_name(self) -> str: ...

    def fit(self, documents: list[str]) -> "EmbeddingModel": ...

    def embed_documents(self, documents: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def save(self, path: str) -> None: ...
