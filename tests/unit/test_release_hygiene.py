from pathlib import Path

from hireflow import __version__


ROOT = Path(__file__).resolve().parents[2]


def test_release_version_is_1_0_0() -> None:
    assert __version__ == "1.0.0"


def test_private_data_directories_contain_no_release_data() -> None:
    for relative in ["data/raw/jobs", "data/raw/resumes", "data/processed", "artifacts/indexes"]:
        directory = ROOT / relative
        leaked = [p for p in directory.rglob("*") if p.is_file() and p.name != ".gitkeep"]
        assert leaked == []
