"""The reference object: build, validate, freeze and reload a cellular-state reference."""

from __future__ import annotations

import dataclasses
import logging
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import schema
from .backends import get_backend
from .backends.base import MappingBackend
from .utils import MetadataReport, resolve_latent, validate_metadata
from .validation import LODOConfig, ValidationResult, run_validation

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class AtlasReference:
    """
    A labelled multi-donor reference in a fixed latent space.

    Parameters
    ----------
    adata
        Reference cells x genes with donor and state annotations in ``obs``. May be
        ``None`` for a reference restored with :meth:`load`.
    donor_key, state_key
        ``obs`` columns holding the donor identifier and the state label.
    celltype_key
        Optional coarser annotation carried along for reporting.
    latent_key
        ``obsm`` key of a precomputed embedding (for example ``"X_scVI"``). ``None``
        computes a PCA fallback (see :func:`~atlasroute.utils.resolve_latent`).
    backend
        Backend name (``"knn"``) or an instance satisfying :class:`~atlasroute.backends.base.MappingBackend`.
    backend_params
        Constructor arguments when ``backend`` is a name.
    n_pcs
        Components for the PCA fallback.
    min_donors_per_state, min_cells_per_state
        Support thresholds for the ``eligible`` flag in the state table.
    random_state
        Seed for the PCA fallback and random-split validation.

    Examples
    --------
    >>> import atlasroute as ar
    >>> ref = ar.Reference(adata, donor_key="donor_id", state_key="cell_state", latent_key="X_scVI")
    >>> res = ref.fit().leave_one_donor_out()
    >>> res.state_metrics[["n_donors_reference", "median_donor_recall", "lodo_stability"]]
    """

    def __init__(
        self,
        adata: AnnData | None,
        *,
        donor_key: str,
        state_key: str,
        celltype_key: str | None = None,
        latent_key: str | None = None,
        backend: str | MappingBackend = "knn",
        backend_params: dict[str, Any] | None = None,
        n_pcs: int = 50,
        min_donors_per_state: int = 3,
        min_cells_per_state: int = 100,
        random_state: int = 0,
    ) -> None:
        self.adata = adata
        self.donor_key = donor_key
        self.state_key = state_key
        self.celltype_key = celltype_key
        self.latent_key = latent_key
        self.n_pcs = int(n_pcs)
        self.min_donors_per_state = int(min_donors_per_state)
        self.min_cells_per_state = int(min_cells_per_state)
        self.random_state = int(random_state)
        if isinstance(backend, str):
            self._backend_template = get_backend(backend, **(backend_params or {}))
        else:
            if not isinstance(backend, MappingBackend):
                raise TypeError("backend must be a name or an object implementing MappingBackend")
            self._backend_template = backend

        # populated by fit() or load()
        self.latent_: np.ndarray | None = None
        self.states_: np.ndarray | None = None
        self.donors_: np.ndarray | None = None
        self.celltypes_: np.ndarray | None = None
        self.obs_names_: np.ndarray | None = None
        self.state_codes_: np.ndarray | None = None
        self.donor_codes_: np.ndarray | None = None
        self.celltype_codes_: np.ndarray | None = None
        self.backend_: MappingBackend | None = None
        self.metadata_: MetadataReport | None = None
        self.latent_provenance_: dict[str, Any] = {}
        self.validation_: ValidationResult | None = None
        self.reference_id_: str | None = None
        self.is_frozen: bool = False

    # ------------------------------------------------------------------ state
    @property
    def is_fitted(self) -> bool:
        return self.backend_ is not None and self.latent_ is not None

    @property
    def centroids_(self) -> np.ndarray:
        """State means in latent space, shape (n_states, n_dims)."""
        self._check_fitted()
        out = np.zeros((len(self.states_), self.latent_.shape[1]), dtype=np.float32)
        for k in range(len(self.states_)):
            sel = self.state_codes_ == k
            if sel.any():
                out[k] = self.latent_[sel].mean(axis=0)
        return out

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("call fit() first")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if not self.is_fitted:
            return f"<AtlasReference unfitted donor_key={self.donor_key!r} state_key={self.state_key!r}>"
        return (
            f"<AtlasReference {self.latent_.shape[0]} cells, {len(self.donors_)} donors, "
            f"{len(self.states_)} states, {self.latent_.shape[1]}-d latent"
            f"{', frozen' if self.is_frozen else ''}{', validated' if self.validation_ else ''}>"
        )

    # -------------------------------------------------------------------- fit
    def fit(self) -> AtlasReference:
        """
        Validate metadata, resolve the latent space and fit the backend on every cell.

        On a reference restored with :meth:`load` only the backend is refitted, from the
        stored latent coordinates.
        """
        if self.is_frozen and self.latent_ is not None:
            self.backend_ = self._backend_template.clone().fit(
                self.latent_, self.states_[self.state_codes_]
            )
            logger.info("refitted backend on frozen latent (%d cells)", self.latent_.shape[0])
            return self
        if self.adata is None:
            raise RuntimeError(
                "no AnnData attached; construct with adata or use AtlasReference.load()"
            )
        self.metadata_ = validate_metadata(
            self.adata,
            donor_key=self.donor_key,
            state_key=self.state_key,
            celltype_key=self.celltype_key,
            min_donors_per_state=self.min_donors_per_state,
            min_cells_per_state=self.min_cells_per_state,
        )
        self.latent_, self.latent_provenance_ = resolve_latent(
            self.adata, latent_key=self.latent_key, n_pcs=self.n_pcs, random_state=self.random_state
        )
        obs = self.adata.obs
        self.states_ = np.asarray(obs[self.state_key].cat.categories).astype(str)
        self.donors_ = np.asarray(obs[self.donor_key].cat.categories).astype(str)
        self.state_codes_ = np.asarray(obs[self.state_key].cat.codes, dtype=np.int32)
        self.donor_codes_ = np.asarray(obs[self.donor_key].cat.codes, dtype=np.int32)
        if self.celltype_key is not None:
            self.celltypes_ = np.asarray(obs[self.celltype_key].cat.categories).astype(str)
            self.celltype_codes_ = np.asarray(obs[self.celltype_key].cat.codes, dtype=np.int32)
        self.obs_names_ = np.asarray(obs.index).astype(str)
        self.backend_ = self._backend_template.clone().fit(
            self.latent_, self.states_[self.state_codes_]
        )
        self.reference_id_ = self.reference_id_ or str(uuid.uuid4())
        self.adata.uns[schema.UNS_REFERENCE] = {
            "reference_id": self.reference_id_,
            "donor_key": self.donor_key,
            "state_key": self.state_key,
            "celltype_key": self.celltype_key,
            "latent": dict(self.latent_provenance_),
            "backend": {"name": self.backend_.name, "params": self.backend_.get_params()},
        }
        logger.info(
            "fitted reference: %d cells, %d donors, %d states, %d-d latent",
            self.latent_.shape[0],
            len(self.donors_),
            len(self.states_),
            self.latent_.shape[1],
        )
        return self

    # ------------------------------------------------------------- validation
    def validate(self, *, method: str = "leave_one_donor_out", **kwargs: Any) -> ValidationResult:
        """
        Run donor-aware validation.

        Parameters
        ----------
        method
            ``"leave_one_donor_out"`` (default) or ``"random_split"``.
        **kwargs
            Passed to :meth:`leave_one_donor_out` or :meth:`random_split`.
        """
        if method == "leave_one_donor_out":
            return self.leave_one_donor_out(**kwargs)
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

    def _run(self, method: str, config: LODOConfig, n_splits: int = 5) -> ValidationResult:
        self._check_fitted()
        result = run_validation(
            method=method,
            X=self.latent_,
            state_labels=self.states_[self.state_codes_],
            donor_labels=self.donors_[self.donor_codes_],
            obs_names=self.obs_names_,
            states=self.states_,
            backend=self._backend_template,
            config=config,
            latent_provenance=self.latent_provenance_,
            metadata_state_table=self.metadata_.state_table if self.metadata_ else None,
            n_splits=n_splits,
        )
        if method == "leave_one_donor_out":
            self.validation_ = result
        return result

    def leave_one_donor_out(
        self, *, config: LODOConfig | None = None, **overrides: Any
    ) -> ValidationResult:
        """
        Hold out each donor in turn and predict its cells from the other donors.

        The result is stored on ``self.validation_`` and included by :meth:`freeze`.
        """
        return self._run("leave_one_donor_out", self._config(config, overrides))

    def random_split(
        self, *, n_splits: int = 5, config: LODOConfig | None = None, **overrides: Any
    ) -> ValidationResult:
        """
        Stratified K-fold over cells ignoring donors: the conventional, optimistic control.
        """
        return self._run("random_split", self._config(config, overrides), n_splits=n_splits)

    def state_table(self) -> pd.DataFrame:
        """Donor support per state, joined with LODO metrics when validation has run."""
        self._check_fitted()
        table = self.metadata_.state_table.copy()
        if self.validation_ is not None:
            sm = self.validation_.state_metrics.drop(
                columns=["eligible", "n_donors_reference"], errors="ignore"
            )
            table = table.join(sm, how="left")
        return table

    # ------------------------------------------------------------ freeze/load
    def freeze(
        self, path: str | Path, *, overwrite: bool = False, include_cell_predictions: bool = True
    ) -> Path:
        """
        Write the frozen reference (see :mod:`atlasroute.io`) and mark this object frozen.
        """
        from .io import write_frozen

        self._check_fitted()
        if self.validation_ is None:
            warnings.warn(
                "freezing a reference without leave-one-donor-out validation; "
                "call leave_one_donor_out() first to store donor-level QC alongside it.",
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
    def load(cls, path: str | Path, *, verify: bool = True) -> AtlasReference:
        """
        Restore a frozen reference. Call :meth:`fit` afterwards to rebuild the backend.
        """
        from .io import read_frozen

        b = read_frozen(path, verify=verify)
        m = b.manifest
        ref = cls(
            None,
            donor_key=m["keys"]["donor"],
            state_key=m["keys"]["state"],
            celltype_key=m["keys"].get("celltype"),
            latent_key=m["keys"].get("latent"),
            backend=m["backend"]["name"],
            backend_params=m["backend"]["params"],
            min_donors_per_state=m["thresholds"]["min_donors_per_state"],
            min_cells_per_state=m["thresholds"]["min_cells_per_state"],
        )
        ref.latent_ = b.latent
        ref.states_ = np.asarray(b.states).astype(str)
        ref.donors_ = np.asarray(b.donors).astype(str)
        ref.celltypes_ = np.asarray(b.celltypes).astype(str) if b.celltypes else None
        ref.obs_names_ = b.obs_names
        ref.state_codes_ = b.state_codes
        ref.donor_codes_ = b.donor_codes
        ref.celltype_codes_ = b.celltype_codes if b.celltypes else None
        ref.latent_provenance_ = dict(m.get("latent", {}))
        ref.reference_id_ = m.get("reference_id")
        ref.validation_ = b.validation
        ref.metadata_ = MetadataReport(
            state_table=b.state_table,
            n_donors=int(m["counts"]["n_donors"]),
            n_cells=int(m["counts"]["n_cells"]),
            n_states=int(m["counts"]["n_states"]),
            warnings=(),
        )
        ref.is_frozen = True
        return ref


Reference = AtlasReference
