"""Configuration-driven concept taxonomy for requirement matching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Concept:
    key: str
    label: str
    aliases: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class MatchingTaxonomy:
    concepts: tuple[Concept, ...]
    software_groups: tuple[Concept, ...]
    preferred_criteria: tuple[Concept, ...]
    strong_education_fields: tuple[str, ...]
    related_education_fields: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "MatchingTaxonomy":
        payload: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        def _items(name: str, weighted: bool) -> tuple[Concept, ...]:
            result: list[Concept] = []
            for key, item in payload.get(name, {}).items():
                result.append(
                    Concept(
                        key=key,
                        label=str(item.get("label", key.replace("_", " ").title())),
                        aliases=tuple(str(v).casefold() for v in item.get("aliases", [])),
                        weight=float(item.get("weight", 1.0)) if weighted else 1.0,
                    )
                )
            return tuple(result)

        education = payload.get("education_fields", {})
        return cls(
            concepts=_items("concepts", weighted=True),
            software_groups=_items("software_groups", weighted=False),
            preferred_criteria=_items("preferred_criteria", weighted=False),
            strong_education_fields=tuple(str(v).casefold() for v in education.get("strong", [])),
            related_education_fields=tuple(str(v).casefold() for v in education.get("related", [])),
        )
