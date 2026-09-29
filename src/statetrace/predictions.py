"""
A backend-agnostic container for cell-state predictions.

StateTrace does not map cells itself. Any mapping tool (scPoli, scANVI/scArches, Symphony,
Azimuth/Seurat, CellTypist, a user classifier, or the built-in kNN backend) produces labels,
optionally probabilities and a native confidence or uncertainty score; :class:`Predictions`
normalises those into one object that the validation, transport and robustness layers consume.
Each ``from_*`` constructor documents where the tool writes its outputs and what is used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import schema

if TYPE_CHECKING:
    from anndata import AnnData


def _series(values: Any, index: pd.Index, name: str, dtype: Any = None) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values.reindex(index)
    else:
        arr = np.asarray(values)
        if arr.shape[0] != len(index):
            raise ValueError(f"{name} has {arr.shape[0]} values for {len(index)} cells")
        s = pd.Series(arr, index=index)
    s.name = name
    return s.astype(dtype) if dtype is not None else s


@dataclass
class Predictions:
    """
    Per-cell state predictions from any backend.

    Attributes
    ----------
    labels
        Predicted state per cell (index = cell names, values = str).
    proba
        Optional cells x states probabilities (columns = states, rows sum to ~1).
    confidence
        Score in [0, 1], higher = more confident. Derived from ``proba`` when not supplied.
    uncertainty
        The tool's native uncertainty when it has one (higher = less certain), kept unchanged.
    support
        Optional local reference support in [0, 1] (neighbour vote fraction, mapping score).
    backend
        Name of the tool that produced the predictions.
    """

    labels: pd.Series
    proba: pd.DataFrame | None = None
    confidence: pd.Series | None = None
    uncertainty: pd.Series | None = None
    support: pd.Series | None = None
    backend: str = "custom"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = pd.Index(self.labels.index.astype(str), name="cell")
        self.labels = pd.Series(self.labels.astype(str).to_numpy(), index=idx, name="label")
        if self.proba is not None:
            p = self.proba.copy()
            p.index = idx
            p.columns = [str(c) for c in p.columns]
            self.proba = p.astype(np.float64)
        if self.confidence is None:
            if self.proba is not None:
                self.confidence = pd.Series(self.proba.max(axis=1).to_numpy(), index=idx)
            elif self.uncertainty is not None:
                u = _series(self.uncertainty, idx, "uncertainty", float)
                if u.min() < 0 or u.max() > 1:
                    warnings.warn(
                        f"{self.backend}: uncertainty is not in [0, 1]; confidence is 1 - min-max scaled uncertainty.",
                        UserWarning,
                        stacklevel=3,
                    )
                    rng = u.max() - u.min()
                    u = (u - u.min()) / rng if rng > 0 else u * 0
                self.confidence = 1.0 - u
            else:
                self.confidence = pd.Series(np.nan, index=idx)
        self.confidence = _series(self.confidence, idx, "confidence", float)
        if self.uncertainty is not None:
            self.uncertainty = _series(self.uncertainty, idx, "uncertainty", float)
        if self.support is not None:
            self.support = _series(self.support, idx, "support", float)

    # ------------------------------------------------------------ properties
    @property
    def obs_names(self) -> pd.Index:
        return self.labels.index

    @property
    def states(self) -> list[str]:
        if self.proba is not None:
            return list(self.proba.columns)
        return sorted(s for s in self.labels.unique() if s != schema.UNKNOWN_LABEL)

    def __len__(self) -> int:
        return len(self.labels)

    # ----------------------------------------------------------- derivatives
    def aligned_proba(self, states: list[str] | np.ndarray) -> np.ndarray:
        """Probabilities re-indexed to ``states`` (zero for states the backend never predicts)."""
        states = [str(s) for s in states]
        if self.proba is not None:
            return self.proba.reindex(columns=states, fill_value=0.0).to_numpy()
        # one-hot from labels weighted by confidence when no matrix is available
        P = np.zeros((len(self), len(states)))
        pos = {s: i for i, s in enumerate(states)}
        conf = self.confidence.fillna(1.0).to_numpy()
        for i, lab in enumerate(self.labels.to_numpy()):
            j = pos.get(lab)
            if j is not None:
                P[i, j] = conf[i]
        return P

    def second_best(self) -> pd.Series:
        if self.proba is None or self.proba.shape[1] < 2:
            return pd.Series(pd.NA, index=self.obs_names, dtype="object")
        order = np.argsort(-self.proba.to_numpy(), axis=1, kind="stable")
        return pd.Series(np.asarray(self.proba.columns)[order[:, 1]], index=self.obs_names)

    def margin(self) -> pd.Series:
        if self.proba is None or self.proba.shape[1] < 2:
            return pd.Series(np.nan, index=self.obs_names)
        srt = -np.sort(-self.proba.to_numpy(), axis=1)
        return pd.Series(srt[:, 0] - srt[:, 1], index=self.obs_names)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "label": self.labels,
                "confidence": self.confidence,
                "margin": self.margin(),
                "second_best": self.second_best(),
            }
        )
        df["uncertainty"] = self.uncertainty if self.uncertainty is not None else np.nan
        df["support"] = self.support if self.support is not None else np.nan
        df["backend"] = self.backend
        return df

    def subset(self, index: pd.Index | np.ndarray) -> Predictions:
        idx = pd.Index(np.asarray(index).astype(str))
        return Predictions(
            labels=self.labels.reindex(idx),
            proba=self.proba.reindex(idx) if self.proba is not None else None,
            confidence=self.confidence.reindex(idx),
            uncertainty=self.uncertainty.reindex(idx) if self.uncertainty is not None else None,
            support=self.support.reindex(idx) if self.support is not None else None,
            backend=self.backend,
            meta=dict(self.meta),
        )

    # --------------------------------------------------------- constructors
    @classmethod
    def from_arrays(
        cls,
        obs_names: Any,
        labels: Any,
        *,
        proba: np.ndarray | pd.DataFrame | None = None,
        states: list[str] | None = None,
        confidence: Any = None,
        uncertainty: Any = None,
        support: Any = None,
        backend: str = "custom",
    ) -> Predictions:
        idx = pd.Index(np.asarray(obs_names).astype(str), name="cell")
        P = None
        if proba is not None:
            if isinstance(proba, pd.DataFrame):
                P = proba
            else:
                if states is None:
                    raise ValueError("states is required when proba is an array")
                P = pd.DataFrame(np.asarray(proba), index=idx, columns=[str(s) for s in states])
        return cls(
            labels=pd.Series(np.asarray(labels).astype(str), index=idx),
            proba=P,
            confidence=None
            if confidence is None
            else _series(confidence, idx, "confidence", float),
            uncertainty=None
            if uncertainty is None
            else _series(uncertainty, idx, "uncertainty", float),
            support=None if support is None else _series(support, idx, "support", float),
            backend=backend,
        )

    @classmethod
    def from_obs(
        cls,
        adata: AnnData,
        label_key: str,
        *,
        proba_key: str | None = None,
        confidence_key: str | None = None,
        uncertainty_key: str | None = None,
        support_key: str | None = None,
        backend: str = "custom",
    ) -> Predictions:
        """
        Build from ``adata.obs`` columns and an optional ``obsm`` probability matrix.

        ``proba_key`` names an ``obsm`` entry; its column names are read from
        ``adata.uns[f"{proba_key}_columns"]`` when the entry is a bare array, or from the
        DataFrame itself.
        """
        for k in (label_key, confidence_key, uncertainty_key, support_key):
            if k is not None and k not in adata.obs:
                raise KeyError(f"obs[{k!r}] not found")
        P = None
        if proba_key is not None:
            if proba_key not in adata.obsm:
                raise KeyError(f"obsm[{proba_key!r}] not found")
            m = adata.obsm[proba_key]
            if isinstance(m, pd.DataFrame):
                P = m.copy()
                P.index = adata.obs_names
            else:
                cols = adata.uns.get(f"{proba_key}_columns")
                if cols is None:
                    raise KeyError(
                        f"uns[{proba_key + '_columns'!r}] must hold the state names of obsm[{proba_key!r}]"
                    )
                P = pd.DataFrame(
                    np.asarray(m), index=adata.obs_names, columns=[str(c) for c in cols]
                )
        return cls.from_arrays(
            adata.obs_names,
            adata.obs[label_key].astype(str).to_numpy(),
            proba=P,
            confidence=None if confidence_key is None else adata.obs[confidence_key].to_numpy(),
            uncertainty=None if uncertainty_key is None else adata.obs[uncertainty_key].to_numpy(),
            support=None if support_key is None else adata.obs[support_key].to_numpy(),
            backend=backend,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        label_col: str,
        *,
        proba_cols: list[str] | None = None,
        confidence_col: str | None = None,
        uncertainty_col: str | None = None,
        support_col: str | None = None,
        backend: str = "custom",
    ) -> Predictions:
        """Build from a table indexed by cell name (for example a CSV exported from R)."""
        P = df[proba_cols].copy() if proba_cols else None
        return cls.from_arrays(
            df.index,
            df[label_col].to_numpy(),
            proba=P,
            confidence=None if confidence_col is None else df[confidence_col].to_numpy(),
            uncertainty=None if uncertainty_col is None else df[uncertainty_col].to_numpy(),
            support=None if support_col is None else df[support_col].to_numpy(),
            backend=backend,
        )

    # tool-specific conventions -------------------------------------------------
    @classmethod
    def from_scpoli(
        cls,
        adata: AnnData,
        *,
        label_key: str = "cell_type_pred",
        uncertainty_key: str | None = "cell_type_uncert",
    ) -> Predictions:
        """
        scPoli (scarches ``scPoli.classify``) writes the predicted label and a per-cell
        uncertainty (0 = certain) into ``obs``; the label key follows the ``cell_type_keys``
        given at training with the ``_pred`` / ``_uncert`` suffixes.
        """
        return cls.from_obs(adata, label_key, uncertainty_key=uncertainty_key, backend="scpoli")

    @classmethod
    def from_scanvi(
        cls,
        adata: AnnData,
        *,
        label_key: str = "C_scANVI",
        proba: pd.DataFrame | None = None,
        proba_key: str | None = None,
    ) -> Predictions:
        """
        scANVI / scArches: ``model.predict()`` gives labels (often stored as ``obs["C_scANVI"]``)
        and ``model.predict(soft=True)`` a cells x states DataFrame; pass the latter as ``proba``
        or store it in ``obsm`` and name it with ``proba_key``.
        """
        p = cls.from_obs(adata, label_key, proba_key=proba_key, backend="scanvi")
        if proba is not None:
            p = cls(labels=p.labels, proba=proba, backend="scanvi")
        return p

    @classmethod
    def from_celltypist(
        cls,
        adata: AnnData,
        *,
        label_key: str = "predicted_labels",
        confidence_key: str | None = "conf_score",
        proba: pd.DataFrame | None = None,
    ) -> Predictions:
        """
        CellTypist ``annotate(...).to_adata()`` writes ``predicted_labels``, ``majority_voting``
        and ``conf_score`` to ``obs``; ``AnnotationResult.probability_matrix`` can be passed as
        ``proba``.
        """
        p = cls.from_obs(adata, label_key, confidence_key=confidence_key, backend="celltypist")
        if proba is not None:
            p = cls(labels=p.labels, proba=proba, confidence=p.confidence, backend="celltypist")
        return p

    @classmethod
    def from_symphony(
        cls,
        df: pd.DataFrame,
        *,
        label_col: str = "cell_type_pred_knn",
        confidence_col: str | None = "cell_type_pred_knn_prob",
        mapping_score_col: str | None = None,
    ) -> Predictions:
        """
        Symphony (R): ``knnPredict`` adds ``<label>_pred_knn`` and ``<label>_pred_knn_prob`` to the
        query metadata; ``mapQuery`` mapping scores (``mapping.score``) become ``support``.
        Export the metadata to CSV indexed by cell and pass it here.
        """
        return cls.from_dataframe(
            df,
            label_col,
            confidence_col=confidence_col,
            support_col=mapping_score_col,
            backend="symphony",
        )

    @classmethod
    def from_azimuth(
        cls,
        df: pd.DataFrame,
        *,
        label_col: str = "predicted.celltype.l2",
        confidence_col: str | None = "predicted.celltype.l2.score",
        mapping_score_col: str | None = "mapping.score",
    ) -> Predictions:
        """
        Azimuth / Seurat ``MapQuery``: ``predicted.<level>`` labels, ``predicted.<level>.score``
        prediction scores and ``mapping.score``; export the metadata to CSV indexed by cell.
        """
        return cls.from_dataframe(
            df,
            label_col,
            confidence_col=confidence_col,
            support_col=mapping_score_col,
            backend="azimuth",
        )
