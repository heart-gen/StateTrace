"""The reference object: build, validate by group, freeze and reload a cellular-state reference."""

from __future__ import annotations

import dataclasses
import logging
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import novelty, schema
from .backends import get_backend
from .backends.base import AnnDataBackend, MappingBackend
from .expression import (
    ExpressionConfig,
    ExpressionSpace,
    build_expression_space,
    cell_coordinates,
)
from .utils import MetadataReport, resolve_latent, validate_metadata
from .validation import LODOConfig, ValidationResult, run_validation

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class StateReference:
    """
    A labelled multi-group reference of cellular states.

    Parameters
    ----------
    adata
        Reference cells x genes with group (donor) and state annotations in ``obs``; ``None``
        for a reference restored with :meth:`load`.
    group_key
        ``obs`` column holding the replication unit (donor, individual). ``donor_key`` is an
        accepted alias.
    state_key
        ``obs`` column holding the state label.
    celltype_key
        Optional coarser annotation carried along for reporting.
    study_key
        Optional ``obs`` column for leave-one-study-out validation.
    latent_key
        ``obsm`` key of a precomputed embedding (``"X_scVI"``). ``None`` computes a PCA fallback.
    backend
        ``"knn"`` (default), a :class:`~statetrace.backends.base.MappingBackend` instance, or an
        :class:`~statetrace.backends.base.AnnDataBackend` that refits an external mapper per fold.
    backend_params
        Constructor arguments when ``backend`` is a name.
    min_groups_per_state, min_cells_per_state
        Support thresholds for the ``eligible`` flag in the state table.
    expression
        :class:`~statetrace.expression.ExpressionConfig` for the frozen expression space
        (donor x state pseudobulk shift diagnostics, see :mod:`statetrace.expression`), built
        from ``adata.X`` or ``layers[expression.layer]`` at :meth:`fit`. ``True`` (default)
        builds it with default settings whenever the reference has genes; ``False`` skips it,
        in which case shift diagnostics fall back to the mapping latent.

    Examples
    --------
    >>> import statetrace as st
    >>> ref = st.Reference(adata, group_key="donor_id", state_key="cell_state", latent_key="X_scVI").fit()
    >>> res = ref.leave_one_group_out()
    >>> res.state_metrics[["n_groups_reference", "median_group_recall", "reproducibility"]]
    """

    def __init__(
        self,
        adata: AnnData | None,
        *,
        group_key: str | None = None,
        donor_key: str | None = None,
        state_key: str,
        celltype_key: str | None = None,
        study_key: str | None = None,
        latent_key: str | None = None,
        backend: str | MappingBackend | AnnDataBackend = "knn",
        backend_params: dict[str, Any] | None = None,
        n_pcs: int = 50,
        min_groups_per_state: int = 3,
        min_cells_per_state: int = 100,
        random_state: int = 0,
        expression: ExpressionConfig | bool = True,
    ) -> None:
        if group_key is None and donor_key is None:
            raise TypeError("group_key (or its alias donor_key) is required")
        if group_key is not None and donor_key is not None and group_key != donor_key:
            raise TypeError("pass either group_key or donor_key, not both")
        self.adata = adata
        self.group_key = group_key or donor_key
        self.state_key = state_key
        self.celltype_key = celltype_key
        self.study_key = study_key
        self.latent_key = latent_key
        self.n_pcs = int(n_pcs)
        self.min_groups_per_state = int(min_groups_per_state)
        self.min_cells_per_state = int(min_cells_per_state)
        self.random_state = int(random_state)
        if isinstance(expression, bool):
            self.expression_config: ExpressionConfig | None = (
                ExpressionConfig() if expression else None
            )
        elif isinstance(expression, ExpressionConfig):
            self.expression_config = expression
        else:
            raise TypeError("expression must be an ExpressionConfig or a bool")
        if isinstance(backend, str):
            self._backend_template = get_backend(backend, **(backend_params or {}))
        elif isinstance(backend, MappingBackend | AnnDataBackend):
            self._backend_template = backend
        else:
            raise TypeError(
                "backend must be a name or an object implementing MappingBackend or AnnDataBackend"
            )

        self.latent_: np.ndarray | None = None
        self.states_: np.ndarray | None = None
        self.groups_: np.ndarray | None = None
        self.studies_: np.ndarray | None = None
        self.celltypes_: np.ndarray | None = None
        self.obs_names_: np.ndarray | None = None
        self.state_codes_: np.ndarray | None = None
        self.group_codes_: np.ndarray | None = None
        self.study_codes_: np.ndarray | None = None
        self.celltype_codes_: np.ndarray | None = None
        self.backend_: Any = None
        self.metadata_: MetadataReport | None = None
        self.latent_provenance_: dict[str, Any] = {}
        self.validation_: ValidationResult | None = None
        self.reference_id_: str | None = None
        self.is_frozen: bool = False
        self.expression_: ExpressionSpace | None = None
        self._centroids: np.ndarray | None = None
        self._geometry: np.ndarray | None = None

    # ------------------------------------------------------------------ state
    @property
    def donor_key(self) -> str:
        return self.group_key

    @property
    def donors_(self) -> np.ndarray | None:
        return self.groups_

    @property
    def is_fitted(self) -> bool:
        return self.backend_ is not None and self.latent_ is not None

    @property
    def uses_anndata_backend(self) -> bool:
        return isinstance(self._backend_template, AnnDataBackend) and not isinstance(
            self._backend_template, MappingBackend
        )

    def _ensure_geometry(self) -> None:
        if self._centroids is None:
            self._centroids, self._geometry = novelty.state_geometry(
                self.latent_, self.state_codes_, len(self.states_)
            )

    @property
    def centroids_(self) -> np.ndarray:
        """State means in latent space, shape (n_states, n_dims)."""
        self._check_latent()
        self._ensure_geometry()
        return self._centroids

    @property
    def geometry_(self) -> np.ndarray:
        """Per-state (RMS radius, mean distance, sd of distance) of reference cells to their centroid."""
        self._check_latent()
        self._ensure_geometry()
        return self._geometry

    @property
    def distance_calibration_(self) -> pd.DataFrame | None:
        """Held-out distance statistics per state from grouped validation, or ``None``."""
        return self.validation_.distance_calibration if self.validation_ is not None else None

    @property
    def has_expression(self) -> bool:
        """Whether a frozen expression space travels with this reference."""
        return self.expression_ is not None

    @property
    def spread_(self) -> np.ndarray:
        """Per-state RMS distance of reference cells to their centroid."""
        return self.geometry_[:, 0]

    def _check_latent(self) -> None:
        if self.latent_ is None:
            raise RuntimeError("call fit() first")

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("call fit() first")

    def __repr__(self) -> str:  # pragma: no cover
        if self.latent_ is None:
            return f"<StateReference unfitted group_key={self.group_key!r} state_key={self.state_key!r}>"
        return (
            f"<StateReference {self.latent_.shape[0]} cells, {len(self.groups_)} groups, "
            f"{len(self.states_)} states, {self.latent_.shape[1]}-d latent"
            f"{', expression' if self.expression_ is not None else ''}"
            f"{', frozen' if self.is_frozen else ''}{', validated' if self.validation_ else ''}>"
        )

    # -------------------------------------------------------------------- fit
    def fit(self) -> StateReference:
        """
        Validate metadata, resolve the latent space and fit the backend on every cell.

        On a reference restored with :meth:`load` only the backend is refitted from the stored
        latent coordinates.
        """
        if self.is_frozen and self.latent_ is not None:
            if self.uses_anndata_backend:
                raise RuntimeError(
                    "a frozen reference has no AnnData; use the built-in kNN backend or pass predictions"
                )
            self.backend_ = self._backend_template.clone().fit(
                self.latent_, self.states_[self.state_codes_]
            )
            return self
        if self.adata is None:
            raise RuntimeError(
                "no AnnData attached; construct with adata or use StateReference.load()"
            )
        self.metadata_ = validate_metadata(
            self.adata,
            group_key=self.group_key,
            state_key=self.state_key,
            celltype_key=self.celltype_key,
            study_key=self.study_key,
            min_groups_per_state=self.min_groups_per_state,
            min_cells_per_state=self.min_cells_per_state,
        )
        self.latent_, self.latent_provenance_ = resolve_latent(
            self.adata, latent_key=self.latent_key, n_pcs=self.n_pcs, random_state=self.random_state
        )
        obs = self.adata.obs
        self.states_ = np.asarray(obs[self.state_key].cat.categories).astype(str)
        self.groups_ = np.asarray(obs[self.group_key].cat.categories).astype(str)
        self.state_codes_ = np.asarray(obs[self.state_key].cat.codes, dtype=np.int32)
        self.group_codes_ = np.asarray(obs[self.group_key].cat.codes, dtype=np.int32)
        if self.study_key is not None:
            self.studies_ = np.asarray(obs[self.study_key].cat.categories).astype(str)
            self.study_codes_ = np.asarray(obs[self.study_key].cat.codes, dtype=np.int32)
        if self.celltype_key is not None:
            self.celltypes_ = np.asarray(obs[self.celltype_key].cat.categories).astype(str)
            self.celltype_codes_ = np.asarray(obs[self.celltype_key].cat.codes, dtype=np.int32)
        self.obs_names_ = np.asarray(obs.index).astype(str)
        self._centroids = None
        self.expression_ = None
        if self.expression_config is not None and self.adata.n_vars > 0:
            X = (
                self.adata.layers[self.expression_config.layer]
                if self.expression_config.layer is not None
                else self.adata.X
            )
            try:
                self.expression_ = build_expression_space(
                    X,
                    self.adata.var_names,
                    self.group_codes_,
                    self.state_codes_,
                    self.states_,
                    config=self.expression_config,
                )
            except ValueError as err:
                warnings.warn(
                    f"expression space not built ({err}); shift diagnostics will use the "
                    "mapping latent. Pass expression=ExpressionConfig(min_cells=...) to adjust.",
                    UserWarning,
                    stacklevel=2,
                )
        if self.uses_anndata_backend:
            self.backend_ = self._backend_template.clone().fit(self.adata, state_key=self.state_key)
        else:
            self.backend_ = self._backend_template.clone().fit(
                self.latent_, self.states_[self.state_codes_]
            )
        self.reference_id_ = self.reference_id_ or str(uuid.uuid4())
        self.adata.uns[schema.UNS_REFERENCE] = {
            "reference_id": self.reference_id_,
            "group_key": self.group_key,
            "state_key": self.state_key,
            "celltype_key": self.celltype_key,
            "study_key": self.study_key,
            "latent": dict(self.latent_provenance_),
            "backend": {"name": self.backend_.name, "params": self.backend_.get_params()},
            "expression": dict(self.expression_.params) if self.expression_ is not None else None,
        }
        logger.info(
            "fitted reference: %d cells, %d groups, %d states, %d-d latent",
            self.latent_.shape[0],
            len(self.groups_),
            len(self.states_),
            self.latent_.shape[1],
        )
        return self

    # ------------------------------------------------------------- validation
    def validate(self, *, method: str = "leave_one_group_out", **kwargs: Any) -> ValidationResult:
        """Dispatch to :meth:`leave_one_group_out`, :meth:`leave_one_study_out` or :meth:`random_split`."""
        if method in ("leave_one_group_out", "leave_one_donor_out"):
            return self.leave_one_group_out(**kwargs)
        if method == "leave_one_study_out":
            return self.leave_one_study_out(**kwargs)
        if method == "random_split":
            return self.random_split(**kwargs)
        raise ValueError(
            f"unknown validation method {method!r}; choose from {schema.VALIDATION_METHODS}"
        )

    def _config(self, config: LODOConfig | None, overrides: dict[str, Any]) -> LODOConfig:
        cfg = config or LODOConfig(random_state=self.random_state)
        if overrides:
            bad = set(overrides) - {f.name for f in dataclasses.fields(LODOConfig)}
            if bad:
                raise TypeError(f"unknown LODOConfig option(s): {sorted(bad)}")
            cfg = dataclasses.replace(cfg, **overrides)
        return cfg

    def _expression_coords(self) -> np.ndarray:
        """Every reference cell in the frozen cell-level expression space."""
        if self.adata is None:
            raise ValueError(
                "reproducibility_space='expression' needs the reference AnnData; a loaded "
                "frozen reference keeps the call it was frozen with"
            )
        if self.expression_ is None or not self.expression_.has_cell_space:
            raise ValueError(
                "reproducibility_space='expression' needs an expression space with its "
                "cell-level part (ExpressionConfig.cell_sample_per_state > 0)"
            )
        layer = self.expression_config.layer if self.expression_config is not None else None
        X = self.adata.layers[layer] if layer is not None else self.adata.X
        return cell_coordinates(self.expression_, X, self.adata.var_names)

    def _run(
        self,
        method: str,
        group_labels: np.ndarray,
        group_key: str,
        config: LODOConfig,
        n_splits: int = 5,
    ) -> ValidationResult:
        self._check_fitted()
        coords = self._expression_coords() if config.reproducibility_space == "expression" else None
        return run_validation(
            method=method,
            group_key=group_key,
            X=None if self.uses_anndata_backend else self.latent_,
            state_labels=self.states_[self.state_codes_],
            group_labels=group_labels,
            obs_names=self.obs_names_,
            states=self.states_,
            backend=self._backend_template,
            config=config,
            latent_provenance=self.latent_provenance_
            if not self.uses_anndata_backend
            else {"source": "external"},
            metadata_state_table=self.metadata_.state_table if self.metadata_ else None,
            n_splits=n_splits,
            adata=self.adata if self.uses_anndata_backend else None,
            state_key=self.state_key,
            study_labels=self.studies_[self.study_codes_] if self.studies_ is not None else None,
            expression_coords=coords,
        )

    def leave_one_group_out(
        self, *, config: LODOConfig | None = None, **overrides: Any
    ) -> ValidationResult:
        """Hold out each group (donor) in turn; the result is stored on ``validation_`` and frozen with the reference."""
        self._check_fitted()
        res = self._run(
            "leave_one_group_out",
            self.groups_[self.group_codes_],
            self.group_key,
            self._config(config, overrides),
        )
        self.validation_ = res
        return res

    leave_one_donor_out = leave_one_group_out

    def leave_one_study_out(
        self, *, config: LODOConfig | None = None, **overrides: Any
    ) -> ValidationResult:
        """Hold out each study (``study_key``) in turn: the harder, cohort-level holdout."""
        self._check_fitted()
        if self.study_key is None or self.studies_ is None:
            raise ValueError("leave_one_study_out needs study_key at construction")
        return self._run(
            "leave_one_study_out",
            self.studies_[self.study_codes_],
            self.study_key,
            self._config(config, overrides),
        )

    def random_split(
        self, *, n_splits: int = 5, config: LODOConfig | None = None, **overrides: Any
    ) -> ValidationResult:
        """Stratified K-fold over cells ignoring groups: the conventional, optimistic control."""
        self._check_fitted()
        return self._run(
            "random_split",
            self.groups_[self.group_codes_],
            self.group_key,
            self._config(config, overrides),
            n_splits=n_splits,
        )

    def state_table(self) -> pd.DataFrame:
        """Group support per state, joined with grouped-holdout metrics when validation has run."""
        self._check_latent()
        table = self.metadata_.state_table.copy()
        if self.validation_ is not None:
            sm = self.validation_.state_metrics.drop(
                columns=["eligible", "n_groups_reference"], errors="ignore"
            )
            table = table.join(sm, how="left")
        return table

    # ------------------------------------------------------------ freeze/load
    def freeze(
        self, path: str | Path, *, overwrite: bool = False, include_cell_predictions: bool = True
    ) -> Path:
        """Write the frozen reference (see :mod:`statetrace.io`) and mark this object frozen."""
        from .io import write_frozen

        self._check_latent()
        if self.validation_ is None:
            warnings.warn(
                "freezing a reference without grouped validation; call leave_one_group_out() first "
                "so group-level reproducibility travels with it.",
                UserWarning,
                stacklevel=2,
            )
        target = write_frozen(
            self, path, overwrite=overwrite, include_cell_predictions=include_cell_predictions
        )
        self.is_frozen = True
        logger.info("froze reference to %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> StateReference:
        """Restore a frozen reference; call :meth:`fit` afterwards to rebuild the kNN backend."""
        from .io import read_frozen

        b = read_frozen(path, verify=verify)
        m = b.manifest
        ref = cls(
            None,
            group_key=m["keys"]["group"],
            state_key=m["keys"]["state"],
            celltype_key=m["keys"].get("celltype"),
            study_key=m["keys"].get("study"),
            latent_key=m["keys"].get("latent"),
            backend=m["backend"]["name"] if m["backend"]["name"] in ("knn",) else "knn",
            backend_params=m["backend"]["params"] if m["backend"]["name"] == "knn" else None,
            min_groups_per_state=m["thresholds"]["min_groups_per_state"],
            min_cells_per_state=m["thresholds"]["min_cells_per_state"],
            expression=False,
        )
        ref.latent_ = b.latent
        ref.expression_ = b.expression
        ref.states_ = np.asarray(b.states).astype(str)
        ref.groups_ = np.asarray(b.groups).astype(str)
        ref.studies_ = np.asarray(b.studies).astype(str) if b.studies else None
        ref.celltypes_ = np.asarray(b.celltypes).astype(str) if b.celltypes else None
        ref.obs_names_ = b.obs_names
        ref.state_codes_ = b.state_codes
        ref.group_codes_ = b.group_codes
        ref.study_codes_ = b.study_codes if b.studies else None
        ref.celltype_codes_ = b.celltype_codes if b.celltypes else None
        ref._centroids, ref._geometry = b.centroids, b.geometry
        ref.latent_provenance_ = dict(m.get("latent", {}))
        ref.reference_id_ = m.get("reference_id")
        ref.validation_ = b.validation
        ref.metadata_ = MetadataReport(
            state_table=b.state_table,
            n_groups=int(m["counts"]["n_groups"]),
            n_cells=int(m["counts"]["n_cells"]),
            n_states=int(m["counts"]["n_states"]),
            n_studies=m["counts"].get("n_studies"),
            warnings=(),
        )
        ref.is_frozen = True
        return ref


Reference = StateReference
