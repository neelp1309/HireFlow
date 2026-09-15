#!/usr/bin/env python
"""Build a resume vector index from processed CandidateProfile records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.config import get_settings  # noqa: E402
from hireflow.embeddings import GeminiEmbeddingModel, TfidfEmbeddingModel  # noqa: E402
from hireflow.models import CandidateProfile  # noqa: E402
from hireflow.retrieval import build_vector_index, candidate_documents  # noqa: E402
from hireflow.utils import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "data" / "processed" / "candidates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "indexes" / "baseline")
    parser.add_argument("--document-mode", choices=["full", "section"], default="full")
    parser.add_argument("--embedding", choices=["tfidf", "gemini"], default="tfidf")
    parser.add_argument("--vector-backend", choices=["auto", "faiss", "numpy"], default="auto")
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates, CandidateProfile)
    documents = [doc for candidate in candidates for doc in candidate_documents(candidate, mode=args.document_mode)]
    texts = [doc.text for doc in documents]

    settings = get_settings()
    if args.embedding == "gemini":
        model = GeminiEmbeddingModel(settings.gemini_api_key or "", settings.gemini_embedding_model)
        model.fit(texts)
    else:
        model = TfidfEmbeddingModel().fit(texts)

    vectors = model.embed_documents(texts)
    index = build_vector_index(vectors, documents, backend=args.vector_backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index.save(args.output_dir)

    if args.embedding == "tfidf":
        model.save(args.output_dir / "tfidf_vectorizer.joblib")
    else:
        model.save(args.output_dir / "embedding_model.txt")

    manifest = {
        "embedding_provider": args.embedding,
        "embedding_model": model.model_name,
        "document_mode": args.document_mode,
        "vector_backend": index.backend_name,
        "document_count": len(documents),
        "candidate_count": len(candidates),
        "vector_dimension": int(vectors.shape[1]),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
