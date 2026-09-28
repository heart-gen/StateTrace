# Installation

AtlasRoute supports Python 3.11 to 3.14.

```bash
pip install atlasroute
```

## Extras

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov, hypothesis, ruff | running the test suite |
| `docs` | sphinx, myst-parser, sphinx-rtd-theme | building this documentation |
| `scvi` | scvi-tools (pulls in torch) | the scVI backend, planned for 0.2 |
| `all` | everything above | |

```bash
pip install "atlasroute[dev]"
```

torch is never a direct dependency. If you plan to use the scVI backend, install a torch
build that matches your CPU or GPU stack first, then `pip install "atlasroute[scvi]"`.

## From source

```bash
git clone https://github.com/heart-gen/AtlasRoute.git
cd AtlasRoute
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
# or: mamba env create -f environment.yml
```
