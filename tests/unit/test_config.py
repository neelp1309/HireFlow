import pytest
from pydantic import ValidationError

from hireflow.config import Settings


def test_default_weights_sum_to_one() -> None:
    settings = Settings(_env_file=None)
    total = (
        settings.weight_semantic
        + settings.weight_required_skills
        + settings.weight_experience
        + settings.weight_role
        + settings.weight_preferred
    )
    assert total == pytest.approx(1.0)


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, weight_semantic=0.5)


def test_retrieval_stage_order_is_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, top_k_retrieval=5, top_k_rerank=10)
