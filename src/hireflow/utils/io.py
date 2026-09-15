"""JSON persistence helpers for validated HireFlow models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_json(path: str | Path, model: BaseModel) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, models: list[BaseModel]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for model in models:
            handle.write(model.model_dump_json() + "\n")


def read_json(path: str | Path, model_type: type[T]) -> T:
    return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path, model_type: type[T]) -> list[T]:
    result: list[T] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                result.append(model_type.model_validate_json(line))
    return result


def write_plain_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
