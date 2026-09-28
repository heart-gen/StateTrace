"""Import hygiene: the package imports quietly, stays light, and works without torch/scvi."""

import subprocess
import sys

import pytest

MODULES = [
    "atlasroute",
    "atlasroute.schema",
    "atlasroute.utils",
    "atlasroute.reference",
    "atlasroute.validation",
    "atlasroute.metrics",
    "atlasroute.io",
    "atlasroute.backends",
    "atlasroute.backends.knn",
    "atlasroute.datasets.synthetic",
    "atlasroute.transport",
    "atlasroute.features",
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
        "import atlasroute as ar\n"
        "assert isinstance(ar.__version__, str)\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_top_level_import_is_light():
    code = (
        "import sys, atlasroute\n"
        "heavy = [m for m in ('scanpy', 'matplotlib', 'torch', 'scvi', 'sklearn') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_scvi_backend_points_to_extra():
    code = (
        "import sys; sys.modules['scvi'] = None\n"
        "from atlasroute.backends import get_backend\n"
        "try:\n"
        "    get_backend('scvi')\n"
        "except ImportError as e:\n"
        "    assert 'atlasroute[scvi]' in str(e), str(e)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_public_names_resolve():
    import atlasroute as ar

    for name in ar.__all__:
        assert getattr(ar, name) is not None
    assert "Reference" in dir(ar)
    with pytest.raises(AttributeError):
        _ = ar.not_a_thing
