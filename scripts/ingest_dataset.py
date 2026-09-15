#!/usr/bin/env python
"""Parse raw resume/JD PDFs into validated, private processed artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hireflow.models import CandidateProfile, JobRequirements  # noqa: E402
from hireflow.parsing import parse_job_pdf, parse_resume_pdf  # noqa: E402
from hireflow.utils import write_json, write_jsonl, write_plain_json  # noqa: E402


def run(resumes_dir: Path, job_pdf: Path, output_dir: Path) -> tuple[list[CandidateProfile], JobRequirements]:
    resume_files = sorted(resumes_dir.glob("*.pdf"))
    if not resume_files:
        raise SystemExit(f"No PDF resumes found under {resumes_dir}")

    candidates: list[CandidateProfile] = []
    failures: list[dict[str, str]] = []
    for path in resume_files:
        try:
            candidates.append(parse_resume_pdf(path))
        except Exception as exc:
            failures.append({"file": path.name, "error": str(exc)})

    if not candidates:
        raise SystemExit("All resume parses failed; refusing to continue")

    job = parse_job_pdf(job_pdf)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_json(output_dir / "job.json", job)
    write_plain_json(
        output_dir / "ingestion_report.json",
        {
            "resume_files_found": len(resume_files),
            "resumes_parsed": len(candidates),
            "resume_failures": failures,
            "job_parsed": True,
            "parser_version": "2.0-deterministic",
        },
    )
    return candidates, job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumes-dir", type=Path, required=True)
    parser.add_argument("--job-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    args = parser.parse_args()
    candidates, job = run(args.resumes_dir, args.job_pdf, args.output_dir)
    print(f"Parsed {len(candidates)} resumes and job '{job.title}'.")
    print(f"Processed artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
