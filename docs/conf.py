# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import pathlib
import sys
import tomllib
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

# Project metadata from pyproject (no package import needed)
project = "statetrace"
author = "Kynon J Benjamin"
_meta: dict = {}
try:
    with open(ROOT / "pyproject.toml", "rb") as f:
        _meta = tomllib.load(f).get("project", {})
        project = _meta.get("name", project) or project
        names = [x.get("name", "") for x in _meta.get("authors", []) if isinstance(x, dict)]
        author = ", ".join(n for n in names if n) or author
except Exception:  # pragma: no cover
    pass


def _read_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(project)
    except PackageNotFoundError:
        return _meta.get("version", "0.0.0")


release = version = _read_version()
copyright = f"{datetime.now(UTC).year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
python_use_unqualified_type_names = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
autodoc_mock_imports = ["scvi", "torch"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
_static = pathlib.Path(__file__).resolve().parent / "_static"
_static.mkdir(exist_ok=True)
html_static_path = ["_static"]

nitpicky = False
