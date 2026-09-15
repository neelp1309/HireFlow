"""Synchronize a reusable candidate corpus into a persistent vector index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hireflow.parsing.resume_parser import parse_resume_pdf
from hireflow.production import CorpusIndexManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--index-name", default="candidate_corpus")
    parser.add_argument("--embedding-provider", choices=["tfidf", "gemini"], default="tfidf")
    parser.add_argument("--document-mode", choices=["full", "section"], default="full")
    parser.add_argument("--vector-backend", choices=["auto", "faiss", "numpy"], default="auto")
    args = parser.parse_args()

    paths = sorted(args.resume_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not paths:
        raise SystemExit(f"No PDF resumes found in {args.resume_dir}")
    candidates = [parse_resume_pdf(path) for path in paths]
    synced = CorpusIndexManager().sync(
        args.index_name,
        candidates,
        embedding_provider=args.embedding_provider,
        document_mode=args.document_mode,
        vector_backend=args.vector_backend,
    )
    print(json.dumps(synced.report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
