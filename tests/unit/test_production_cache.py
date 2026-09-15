from pathlib import Path

from hireflow.models import CandidateProfile, JobRequirements
from hireflow.product import ProductSearchConfig
from hireflow.production.cache import MemoryTTLCache, build_search_cache_key


def test_memory_ttl_cache_is_bounded_and_reuses_value() -> None:
    cache = MemoryTTLCache[str](max_entries=2, ttl_seconds=60)
    cache.set("a", "A")
    cache.set("b", "B")
    assert cache.get("a") == "A"
    cache.set("c", "C")
    assert cache.get("b") is None
    assert cache.get("a") == "A"
    assert cache.get("c") == "C"


def test_search_cache_key_invalidates_when_display_identity_changes() -> None:
    job = JobRequirements(job_id="j1", title="Senior Accountant")
    one = CandidateProfile(
        candidate_id="c1",
        source_file=Path("one.pdf"),
        full_name="Alice Example",
        professional_summary="Experienced accountant",
    )
    two = one.model_copy(update={"full_name": "Different Name", "location": "Elsewhere"})
    config = ProductSearchConfig()
    weights = {"semantic": 0.25, "required_skills": 0.3, "experience": 0.15, "role": 0.2, "preferred": 0.1}
    assert build_search_cache_key(job, [one], config, weights) != build_search_cache_key(job, [two], config, weights)
