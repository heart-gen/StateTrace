.PHONY: test test-slow lint format docs release

PY ?= python

test:
	$(PY) -m pytest -q

test-slow:
	$(PY) -m pytest -q --run-slow

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

format:
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

docs:
	$(PY) -m sphinx -W -b html docs docs/_build/html

# Usage: make release v=0.1.1
# Bumps the version, commits and tags; the Release workflow builds and
# publishes when the tag is pushed (after the CI workflow passes).
release: lint test
	poetry version $(v)
	git add pyproject.toml
	git commit -m "Release version v$$(poetry version -s)"
	git tag v$$(poetry version -s)
	@echo "Now: git push origin main --tags"
