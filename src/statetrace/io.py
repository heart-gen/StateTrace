"""
The frozen reference format (``*.statetrace``).

A directory (or zip of the same layout) holding the latent coordinates, labels, categories,
state geometry, state table, provenance manifest and, when validation ran, the validation
tables. Nothing is pickled: arrays are ``.npy``/``.npz`` (read with ``allow_pickle=False``),
tables CSV, metadata JSON.

Layout::

    name.statetrace/
        manifest.json     format_version, versions, keys, provenance, backend, counts, checksums
        latent.npy        float32 (n_cells, n_dims)
        cells.npz         obs_names, state_code, group_code, study_code, celltype_code (-1 when absent)
        categories.json   code -> label lists
        centroids.npy     float32 (n_states, n_dims) state means
        geometry.npy      float32 (n_states, 3) RMS radius, mean and sd of cell-to-centroid distance per state
        state_table.csv   group support per state
        validation/       ValidationResult.to_dir() output (optional)
        expression/       frozen expression space (optional, format 3): genes.json, mu.npy,
                          sigma.npy, n_groups.npy, components.npy, profiles.npy, rows.csv,
                          null.csv, null_stats.csv, params.json; format 4 adds the held-out
                          reference profiles (null_profiles.npy) and the cell-level
                          space: cell_center.npy, cell_scale.npy, cell_components.npy,
                          cell_coords.npy, cell_rows.csv
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
    from .expression import ExpressionSpace
    from .reference import StateReference
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
    group_codes: np.ndarray
    study_codes: np.ndarray
    celltype_codes: np.ndarray
    states: list[str]
    groups: list[str]
    studies: list[str] | None
    celltypes: list[str] | None
    centroids: np.ndarray
    geometry: np.ndarray
    state_table: pd.DataFrame
    validation: ValidationResult | None
    expression: ExpressionSpace | None = None


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


def _codes_or_missing(codes: np.ndarray | None, n: int) -> np.ndarray:
    return (
        np.asarray(codes, dtype=np.int32) if codes is not None else np.full(n, -1, dtype=np.int32)
    )


def write_frozen(
    reference: StateReference,
    path: str | Path,
    *,
    overwrite: bool = False,
    include_cell_predictions: bool = True,
) -> Path:
    """Serialise a fitted :class:`~statetrace.reference.StateReference` to ``path`` (directory, or zip when the path ends in ``.zip``)."""
    from . import __version__

    if reference.latent_ is None:
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
        shutil.rmtree(target) if target.is_dir() else target.unlink()

    n = len(reference.obs_names_)
    tmp_root = Path(tempfile.mkdtemp(prefix="statetrace-freeze-"))
    out = tmp_root / stem
    out.mkdir()
    try:
        np.save(out / "latent.npy", np.asarray(reference.latent_, dtype=np.float32))
        np.savez(
            out / "cells.npz",
            obs_names=np.asarray(reference.obs_names_).astype(str),
            state_code=np.asarray(reference.state_codes_, dtype=np.int32),
            group_code=np.asarray(reference.group_codes_, dtype=np.int32),
            study_code=_codes_or_missing(reference.study_codes_, n),
            celltype_code=_codes_or_missing(reference.celltype_codes_, n),
        )
        (out / "categories.json").write_text(
            json.dumps(
                {
                    "states": [str(s) for s in reference.states_],
                    "groups": [str(g) for g in reference.groups_],
                    "studies": [str(s) for s in reference.studies_]
                    if reference.studies_ is not None
                    else None,
                    "celltypes": [str(c) for c in reference.celltypes_]
                    if reference.celltypes_ is not None
                    else None,
                },
                indent=2,
            )
        )
        np.save(out / "centroids.npy", reference.centroids_.astype(np.float32))
        np.save(out / "geometry.npy", reference.geometry_.astype(np.float32))
        reference.metadata_.state_table.to_csv(out / "state_table.csv")
        if reference.validation_ is not None:
            reference.validation_.to_dir(
                out / "validation", include_cell_predictions=include_cell_predictions
            )
        if reference.expression_ is not None:
            _write_expression(reference.expression_, out / schema.EXPRESSION_DIR)
        files = {str(p.relative_to(out)): _sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
        backend_name = (
            getattr(reference.backend_, "name", "external")
            if reference.backend_ is not None
            else "knn"
        )
        backend_params = reference.backend_.get_params() if reference.backend_ is not None else {}
        manifest = {
            "format_version": schema.FORMAT_VERSION,
            "statetrace_version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "reference_id": reference.reference_id_ or str(uuid.uuid4()),
            "libraries": _library_versions(),
            "keys": {
                "group": reference.group_key,
                "state": reference.state_key,
                "celltype": reference.celltype_key,
                "study": reference.study_key,
                "latent": reference.latent_key,
            },
            "latent": reference.latent_provenance_,
            "backend": {"name": backend_name, "params": backend_params},
            "counts": {
                "n_cells": int(n),
                "n_groups": int(len(reference.groups_)),
                "n_states": int(len(reference.states_)),
                "n_studies": int(len(reference.studies_))
                if reference.studies_ is not None
                else None,
                "n_dims": int(reference.latent_.shape[1]),
            },
            "thresholds": {
                "min_groups_per_state": reference.min_groups_per_state,
                "min_cells_per_state": reference.min_cells_per_state,
            },
            "validated": reference.validation_ is not None,
            "expression": dict(reference.expression_.params)
            if reference.expression_ is not None
            else None,
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


def _write_expression(space: ExpressionSpace, out: Path) -> None:
    out.mkdir()
    (out / "genes.json").write_text(json.dumps([str(g) for g in space.genes]))
    np.save(out / "mu.npy", np.asarray(space.mu, dtype=np.float32))
    np.save(out / "sigma.npy", np.asarray(space.sigma, dtype=np.float32))
    np.save(out / "n_groups.npy", np.asarray(space.n_groups, dtype=np.int32))
    np.save(out / "components.npy", np.asarray(space.components, dtype=np.float32))
    np.save(out / "profiles.npy", np.asarray(space.profiles, dtype=np.float32))
    space.rows.to_csv(out / "rows.csv", index=False)
    space.null.to_csv(out / "null.csv", index=False)
    space.null_stats.to_csv(out / "null_stats.csv")
    (out / "params.json").write_text(json.dumps(space.params, indent=2, default=str))
    if space.null_profiles is not None:
        np.save(out / "null_profiles.npy", np.asarray(space.null_profiles, dtype=np.float32))
    if space.has_cell_space:
        for name in ("cell_center", "cell_scale", "cell_components", "cell_coords"):
            np.save(out / f"{name}.npy", np.asarray(getattr(space, name), dtype=np.float32))
        space.cell_rows.to_csv(out / "cell_rows.csv", index=False)


def _read_expression(src: _Source, states: list[str]) -> ExpressionSpace | None:
    from .expression import ExpressionSpace

    d = schema.EXPRESSION_DIR
    if not src.exists(f"{d}/params.json"):
        return None

    def arr(name: str) -> np.ndarray:
        return np.load(_io.BytesIO(src.read(f"{d}/{name}")), allow_pickle=False)

    null_stats = pd.read_csv(_io.BytesIO(src.read(f"{d}/null_stats.csv")), index_col="state")
    null_stats.index = null_stats.index.astype(str)
    cell: dict[str, Any] = {}
    if src.exists(f"{d}/null_profiles.npy"):  # format 4
        cell["null_profiles"] = arr("null_profiles.npy")
    if src.exists(f"{d}/cell_coords.npy"):  # format 4
        cell.update(
            {
                name: arr(f"{name}.npy")
                for name in ("cell_center", "cell_scale", "cell_components", "cell_coords")
            }
        )
        cell["cell_rows"] = pd.read_csv(_io.BytesIO(src.read(f"{d}/cell_rows.csv")))
    return ExpressionSpace(
        genes=np.asarray(json.loads(src.read(f"{d}/genes.json").decode())).astype(str),
        states=np.asarray(states).astype(str),
        profiles=arr("profiles.npy"),
        rows=pd.read_csv(_io.BytesIO(src.read(f"{d}/rows.csv"))),
        mu=arr("mu.npy"),
        sigma=arr("sigma.npy"),
        n_groups=arr("n_groups.npy"),
        components=arr("components.npy"),
        null=pd.read_csv(_io.BytesIO(src.read(f"{d}/null.csv"))),
        null_stats=null_stats,
        params=json.loads(src.read(f"{d}/params.json").decode()),
        **cell,
    )


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
            manifests = [n for n in self.zf.namelist() if n.endswith(schema.MANIFEST_FILE)]
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
    """Read a frozen reference directory or archive, verifying the format version and every checksum."""
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
                f"{path}: format_version {fv} is newer than this StateTrace supports ({schema.FORMAT_VERSION})"
            )
        if fv < schema.MIN_READABLE_FORMAT_VERSION:
            raise FrozenReferenceError(
                f"{path}: format_version {fv} predates the group/study layout; re-freeze the reference"
            )
        if verify:
            for name, digest in manifest.get("files", {}).items():
                if hashlib.sha256(src.read(name)).hexdigest() != digest:
                    raise FrozenReferenceError(f"{path}: checksum mismatch for {name}")
        latent = np.load(_io.BytesIO(src.read("latent.npy")), allow_pickle=False)
        cells = np.load(_io.BytesIO(src.read("cells.npz")), allow_pickle=False)
        cats = json.loads(src.read("categories.json").decode())
        centroids = np.load(_io.BytesIO(src.read("centroids.npy")), allow_pickle=False)
        geometry = np.load(_io.BytesIO(src.read("geometry.npy")), allow_pickle=False)
        state_table = pd.read_csv(_io.BytesIO(src.read("state_table.csv")), index_col="state")
        state_table.index = state_table.index.astype(str)
        validation = None
        if src.exists("validation/config.json"):
            tmp = Path(tempfile.mkdtemp(prefix="statetrace-read-"))
            try:
                validation = ValidationResult.from_dir(src.extract_dir("validation", tmp))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        states = [str(s) for s in cats["states"]]
        return FrozenBundle(
            manifest=manifest,
            latent=np.asarray(latent, dtype=np.float32),
            obs_names=np.asarray(cells["obs_names"]).astype(str),
            state_codes=np.asarray(cells["state_code"], dtype=np.int32),
            group_codes=np.asarray(cells["group_code"], dtype=np.int32),
            study_codes=np.asarray(cells["study_code"], dtype=np.int32),
            celltype_codes=np.asarray(cells["celltype_code"], dtype=np.int32),
            states=states,
            groups=[str(g) for g in cats["groups"]],
            studies=[str(s) for s in cats["studies"]] if cats.get("studies") else None,
            celltypes=[str(c) for c in cats["celltypes"]] if cats.get("celltypes") else None,
            centroids=np.asarray(centroids, dtype=np.float32),
            geometry=np.asarray(geometry, dtype=np.float32),
            state_table=state_table,
            validation=validation,
            expression=_read_expression(src, states),
        )
    finally:
        src.close()
