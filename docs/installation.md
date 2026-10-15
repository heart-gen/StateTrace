# Installation

StateTrace supports Python 3.11 to 3.14.

```bash
pip install statetrace
```

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov, hypothesis, ruff | running the test suite |
| `docs` | sphinx, myst-parser, sphinx-rtd-theme | building this documentation |
| `plot` | matplotlib | `statetrace.plotting` (planned) |
| `scvi` | scvi-tools (pulls in torch) | the scANVI refit-per-fold backend (planned) |
| `all` | everything above | |

```bash
pip install "statetrace[dev]"
```

torch is never a direct dependency. If you plan to use the scANVI backend, install a torch
build that matches your CPU or GPU stack first, then `pip install "statetrace[scvi]"`.

## From source

```bash
git clone https://github.com/heart-gen/StateTrace.git
cd StateTrace
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
# or: mamba env create -f environment.yml
```
