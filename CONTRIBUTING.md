# Contributing

StateTrace is developed against a gated benchmark (StateTrace-benchmarking): a change that alters
a result on the same data is adopted only after it is chosen on the simulation grid, never on the
real data of the gate it is meant to pass. Please keep that in mind when proposing a change to a
default.

## Set up

```bash
git clone https://github.com/heart-gen/StateTrace.git && cd StateTrace
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
make lint PY=.venv/bin/python && make test PY=.venv/bin/python && make docs PY=.venv/bin/python
```

`make test` runs the fast suite (no torch or scvi-tools needed); `pytest --run-slow` adds the
backend tests. The docs build re-executes the tutorial notebooks and fails on warnings.

## Pull requests

- Branch from `main`; one topic per pull request; CI (ruff, tests on Python 3.11 to 3.14, build,
  docs) must pass.
- Add or adjust tests for behaviour you change; every public object, config field, AnnData key
  (`schema.py`) and frozen-format change is documented under `docs/`.
- Add a line to `CHANGELOG.md` under `[Unreleased]` (Added / Changed / Fixed), stating what a user
  sees differently. A default that changes results on the same data goes under **Changed** with
  the way to reproduce the previous behaviour.
- Commit messages: a subject line and, when useful, a body; no trailers.
- The frozen `.statetrace` format is versioned; a reader must keep loading every earlier version.

## Reporting a problem

Open an issue with the StateTrace version (`statetrace.__version__`), the backend you mapped
with, the shape of the reference and query (`n_obs`, number of donors and states) and the
smallest `obs` table that reproduces it. Do not attach controlled-access data.
