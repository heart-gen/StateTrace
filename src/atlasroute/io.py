"""
The frozen reference format (``*.atlasroute``).

A frozen reference is a directory (or a zip archive of the same layout) holding the
latent coordinates, labels, categories, state table, provenance manifest and, when
validation ran, the validation tables. Nothing is pickled: arrays are ``.npy``/``.npz``
(read with ``allow_pickle=False``), tables are CSV and metadata is JSON, so a reference
frozen today can be read by a future release and inspected without AtlasRoute.

Layout::

    name.atlasroute/
        manifest.json     format_version, versions, keys, provenance, backend, counts, file checksums
        latent.npy        float32 (n_cells, n_dims)
        cells.npz         obs_names, state_code, donor_code, celltype_code (-1 when absent)
        categories.json   code -> label lists
        centroids.npy     float32 (n_states, n_dims) state means in latent space
        state_table.csv   donor support per state
        validation/       ValidationResult.to_dir() output (optional)
"""

from __future__ import annotations

import hashlib
import io as _io
import json
import platform
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import schema

if TYPE_CHECKING:
    from .reference import AtlasReference
    from .validation import ValidationResult


class FrozenReferenceError(RuntimeError):
    """Raised when a frozen reference is missing, corrupt or from a newer format."""


@dataclass
class FrozenBundle:
    """In-memory content of a frozen reference, as returned by :func:`read_frozen`."""

    manifest: dict[str, Any]
    latent: np.ndarray
    obs_names: np.ndarray
    state_codes: np.ndarray
    donor_codes: np.ndarray
    celltype_codes: np.ndarray
    states: list[str]
    donors: list[str]
    celltypes: list[str] | None
    centroids: np.ndarray
    state_table: pd.DataFrame
    validation: ValidationResult | None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "anndata", "scanpy"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover
            out[name] = "not installed"
    return out


def write_frozen(
    reference: AtlasReference,
    path: str | Path,
    *,
    overwrite: bool = False,
    include_cell_predictions: bool = True,
) -> Path:
    """
    Serialise a fitted :class:`AtlasReference` to ``path``.

    ``path`` ending in ``.zip`` produces an archive; otherwise a directory. The
    ``.atlasroute`` suffix is added when missing.
    """
    from . import __version__

    if not reference.is_fitted:
        raise RuntimeError("call fit() before freeze()")
    path = Path(path)
    as_zip = path.suffix == ".zip"
    stem = path.name[: -len(".zip")] if as_zip else path.name
    if not stem.endswith(schema.FROZEN_SUFFIX):
        stem = stem + schema.FROZEN_SUFFIX
    target = path.with_name(stem + ".zip") if as_zip else path.with_name(stem)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    tmp_root = Path(tempfile.mkdtemp(prefix="atlasroute-freeze-"))
    out = tmp_root / stem
    out.mkdir()
    try:
        np.save(out / "latent.npy", np.asarray(reference.latent_, dtype=np.float32))
        np.savez(
            out / "cells.npz",
            obs_names=np.asarray(reference.obs_names_).astype(str),
            state_code=np.asarray(reference.state_codes_, dtype=np.int32),
            donor_code=np.asarray(reference.donor_codes_, dtype=np.int32),
            celltype_code=(
                np.asarray(reference.celltype_codes_, dtype=np.int32)
                if reference.celltype_codes_ is not None
                else np.full(len(reference.obs_names_), -1, dtype=np.int32)
            ),
        )
        (out / "categories.json").write_text(
            json.dumps(
                {
                    "states": [str(s) for s in reference.states_],
                    "donors": [str(d) for d in reference.donors_],
                    "celltypes": (
                        [str(c) for c in reference.celltypes_]
                        if reference.celltypes_ is not None
                        else None
                    ),
                },
                indent=2,
            )
        )
        np.save(out / "centroids.npy", reference.centroids_.astype(np.float32))
        reference.metadata_.state_table.to_csv(out / "state_table.csv")
        if reference.validation_ is not None:
            reference.validation_.to_dir(
                out / "validation", include_cell_predictions=include_cell_predictions
            )
        files = {str(p.relative_to(out)): _sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
        manifest = {
            "format_version": schema.FORMAT_VERSION,
            "atlasroute_version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "reference_id": reference.reference_id_ or str(uuid.uuid4()),
            "libraries": _library_versions(),
            "keys": {
                "donor": reference.donor_key,
                "state": reference.state_key,
                "celltype": reference.celltype_key,
                "latent": reference.latent_key,
            },
            "latent": reference.latent_provenance_,
            "backend": {"name": reference.backend_.name, "params": reference.backend_.get_params()},
            "counts": {
                "n_cells": int(reference.latent_.shape[0]),
                "n_donors": int(len(reference.donors_)),
                "n_states": int(len(reference.states_)),
                "n_dims": int(reference.latent_.shape[1]),
            },
            "thresholds": {
                "min_donors_per_state": reference.min_donors_per_state,
                "min_cells_per_state": reference.min_cells_per_state,
            },
            "validated": reference.validation_ is not None,
            "files": files,
        }
        (out / schema.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        target.parent.mkdir(parents=True, exist_ok=True)
        if as_zip:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(out.rglob("*")):
                    if p.is_file():
                        zf.write(p, arcname=str(Path(stem) / p.relative_to(out)))
        else:
            shutil.move(str(out), str(target))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return target


class _Source:
    """Uniform byte access to a directory or a zip archive."""

    def __init__(self, path: Path):
        self.path = path
        self.zf: zipfile.ZipFile | None = None
        self.prefix = ""
        if path.is_dir():
            return
        if zipfile.is_zipfile(path):
            self.zf = zipfile.ZipFile(path)
            names = self.zf.namelist()
            manifests = [n for n in names if n.endswith(schema.MANIFEST_FILE)]
            if not manifests:
                raise FrozenReferenceError(f"{path}: no {schema.MANIFEST_FILE} in archive")
            self.prefix = manifests[0][: -len(schema.MANIFEST_FILE)]
            return
        raise FrozenReferenceError(f"{path} is neither a directory nor a zip archive")

    def read(self, name: str) -> bytes:
        if self.zf is not None:
            try:
                return self.zf.read(self.prefix + name)
            except KeyError:
                raise FrozenReferenceError(f"{self.path}: missing {name}") from None
        p = self.path / name
        if not p.is_file():
            raise FrozenReferenceError(f"{self.path}: missing {name}")
        return p.read_bytes()

    def exists(self, name: str) -> bool:
        if self.zf is not None:
            return (self.prefix + name) in self.zf.namelist()
        return (self.path / name).exists()

    def extract_dir(self, name: str, dest: Path) -> Path:
        if self.zf is not None:
            for n in self.zf.namelist():
                if n.startswith(self.prefix + name + "/") and not n.endswith("/"):
                    rel = Path(n[len(self.prefix) :])
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    (dest / rel).write_bytes(self.zf.read(n))
            return dest / name
        return self.path / name

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def read_frozen(path: str | Path, *, verify: bool = True) -> FrozenBundle:
    """
    Read a frozen reference directory or archive into memory.

    Parameters
    ----------
    path
        Directory or ``.zip`` written by :func:`write_frozen`.
    verify
        Recompute the SHA-256 of every listed file and fail on any mismatch.
    """
    from .validation import ValidationResult

    path = Path(path)
    if not path.exists():
        raise FrozenReferenceError(f"{path} does not exist")
    src = _Source(path)
    try:
        manifest = json.loads(src.read(schema.MANIFEST_FILE).decode())
        fv = int(manifest.get("format_version", 0))
        if fv > schema.FORMAT_VERSION:
            raise FrozenReferenceError(
                f"{path}: format_version {fv} is newer than this AtlasRoute supports ({schema.FORMAT_VERSION})"
            )
        if verify:
            for name, digest in manifest.get("files", {}).items():
                got = hashlib.sha256(src.read(name)).hexdigest()
                if got != digest:
                    raise FrozenReferenceError(f"{path}: checksum mismatch for {name}")
        latent = np.load(_io.BytesIO(src.read("latent.npy")), allow_pickle=False)
        cells = np.load(_io.BytesIO(src.read("cells.npz")), allow_pickle=False)
        cats = json.loads(src.read("categories.json").decode())
        centroids = np.load(_io.BytesIO(src.read("centroids.npy")), allow_pickle=False)
        state_table = pd.read_csv(_io.BytesIO(src.read("state_table.csv")), index_col="state")
        state_table.index = state_table.index.astype(str)
        validation = None
        if src.exists("validation/config.json"):
            tmp = Path(tempfile.mkdtemp(prefix="atlasroute-read-"))
            try:
                vdir = src.extract_dir("validation", tmp)
                validation = ValidationResult.from_dir(vdir)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return FrozenBundle(
            manifest=manifest,
            latent=np.asarray(latent, dtype=np.float32),
            obs_names=np.asarray(cells["obs_names"]).astype(str),
            state_codes=np.asarray(cells["state_code"], dtype=np.int32),
            donor_codes=np.asarray(cells["donor_code"], dtype=np.int32),
            celltype_codes=np.asarray(cells["celltype_code"], dtype=np.int32),
            states=[str(s) for s in cats["states"]],
            donors=[str(d) for d in cats["donors"]],
            celltypes=[str(c) for c in cats["celltypes"]] if cats.get("celltypes") else None,
            centroids=np.asarray(centroids, dtype=np.float32),
            state_table=state_table,
            validation=validation,
        )
    finally:
        src.close()
