.PHONY: install install-dev lint format format-check typecheck test coverage ci precommit clean

# --- Install ---
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# --- Lint & Format ---
lint:
	ruff check src/ tests/ config/

format:
	ruff format src/ tests/ config/

format-check:
	ruff format --check src/ tests/ config/

typecheck:
	mypy --strict src/

# --- Test ---
test:
	python -m pytest tests/ -v --tb=short

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# --- Full check (CI entry point) ---
ci: lint format-check typecheck test

# --- Pre-commit hook install ---
precommit:
	pre-commit install

# --- Clean ---
clean:
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
