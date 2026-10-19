"""Import hygiene: the package imports quietly, stays light, and works without torch/scvi."""

import subprocess
import sys

import pytest

MODULES = [
    "statetrace",
    "statetrace.schema",
    "statetrace.utils",
    "statetrace.reference",
    "statetrace.validation",
    "statetrace.metrics",
    "statetrace.io",
    "statetrace.backends",
    "statetrace.backends.knn",
    "statetrace.datasets.synthetic",
    "statetrace.predictions",
    "statetrace.transport",
    "statetrace.robustness",
    "statetrace.aggregation",
    "statetrace.features",
    "statetrace.calibration",
    "statetrace.reporting",
    "statetrace.novelty",
]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_modules_import_silently():
    proc = _run("import " + ", ".join(MODULES))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
    assert "Warning" not in proc.stderr, proc.stderr


def test_modules_import_without_torch_or_scvi():
    code = (
        "import sys\n"
        "for m in ('torch', 'torch.cuda', 'scvi', 'scvi_tools'): sys.modules[m] = None\n"
        "import " + ", ".join(MODULES) + "\n"
        "import statetrace as st\n"
        "assert isinstance(st.__version__, str)\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_top_level_import_is_light():
    code = (
        "import sys, statetrace\n"
        "heavy = [m for m in ('scanpy', 'matplotlib', 'torch', 'scvi', 'sklearn') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_scanvi_backend_points_to_extra():
    code = (
        "import sys; sys.modules['scvi'] = None\n"
        "from statetrace.backends import get_backend\n"
        "try:\n    get_backend('scanvi')\n"
        "except ImportError as e:\n    assert 'statetrace-sc[scvi]' in str(e), str(e)\n"
        "else:\n    raise SystemExit('expected ImportError')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_plotting_points_to_extra():
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "from statetrace.plotting import plot_state_reproducibility\n"
        "try:\n    plot_state_reproducibility()\n"
        "except ImportError as e:\n    assert 'statetrace-sc[plot]' in str(e), str(e)\n"
        "else:\n    raise SystemExit('expected ImportError')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_public_names_resolve():
    import statetrace as st

    for name in st.__all__:
        assert getattr(st, name) is not None
    assert "Reference" in dir(st)
    with pytest.raises(AttributeError):
        _ = st.not_a_thing
