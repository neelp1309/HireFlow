.PHONY: install install-dev install-faiss demo validate validate-phase2 validate-phase3 validate-phase4 validate-phase5 validate-phase6 validate-release evaluate rank sync-index test lint typecheck run api docker-build clean

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

install-faiss:
	python -m pip install -r requirements-faiss.txt

demo:
	PYTHONPATH=src python scripts/demo_structured_search.py

validate:
	PYTHONPATH=src python scripts/validate_foundation.py

validate-phase2:
	PYTHONPATH=src python scripts/validate_phase2.py

validate-phase3:
	PYTHONPATH=src python scripts/validate_phase3.py

validate-phase4:
	PYTHONPATH=src python scripts/validate_phase4.py

validate-phase5:
	PYTHONPATH=src python scripts/validate_phase5.py

validate-phase6:
	PYTHONPATH=src python scripts/validate_phase6.py

validate-release:
	PYTHONPATH=src python scripts/validate_release.py

evaluate:
	PYTHONPATH=src python scripts/evaluate_phase4.py

rank:
	PYTHONPATH=src python scripts/rank_candidates.py --evaluator heuristic

sync-index:
	PYTHONPATH=src python scripts/sync_index.py --resume-dir data/raw/resumes --vector-backend auto

test:
	PYTHONPATH=src pytest

lint:
	ruff check src tests scripts

typecheck:
	mypy src

run:
	PYTHONPATH=src streamlit run app/streamlit_app.py

api:
	PYTHONPATH=src uvicorn hireflow.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t hireflow:1.0.0 .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
