"""
Backend robustness: is a transport conclusion the same whichever mapper produced it?

Given the same query mapped by several tools (scPoli, scANVI, Symphony, the built-in kNN,
...), report per-cell agreement, per-state concordance and, when full transport routes are
given, whether the inference-gating status agrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .predictions import Predictions
from .transport import TransportRoute


def consensus_status(
    status: pd.DataFrame, alternative: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Per-state consensus of transport statuses across backends (one column per backend).

    Returns the input columns plus ``status_agreement`` (every backend carries the same
    status), ``consensus`` (the modal status), ``consensus_support`` (fraction of backends
    carrying it: the consensus's reliability), and, when ``alternative`` gives each backend's
    ``status_alternative`` (or ``status_interval_alternative``) per state,
    ``agreement_within_tolerance`` (some label is consistent with every backend, each backend
    accepting its status or its alternative) and ``consensus_within_tolerance`` (that label,
    the modal status when several qualify; ``None`` otherwise).

    A consensus is a sound way to apply StateTrace with several mappers; it is a reading
    across backends with its reliability stated, never a substitute for the per-backend status
    or for a criterion read on exact agreement.
    """
    names = list(status.columns)
    out = status.copy()
    vals = out[names]
    out["status_agreement"] = vals.nunique(axis=1, dropna=True) == 1
    cons, support, within, within_label = [], [], [], []
    for idx, row in vals.iterrows():
        present = row.dropna().astype(str)
        if present.empty:
            cons.append(None)
            support.append(np.nan)
            within.append(False)
            within_label.append(None)
            continue
        counts = present.value_counts()
        cons.append(counts.index[0])
        support.append(float(counts.iloc[0] / len(present)))
        if alternative is None:
            within.append(bool(counts.iloc[0] == len(present)))
            within_label.append(counts.index[0] if counts.iloc[0] == len(present) else None)
            continue
        accept = []
        for b in present.index:
            alt = alternative.loc[idx, b] if (idx in alternative.index and b in alternative.columns) else None
            s = {str(present[b])}
            if isinstance(alt, str) and alt:
                s.add(alt)
            accept.append(s)
        common = set.intersection(*accept) if accept else set()
        within.append(bool(common))
        if not common:
            within_label.append(None)
        else:
            ranked = [c for c in counts.index if c in common]
            within_label.append(ranked[0] if ranked else sorted(common)[0])
    out["consensus"] = cons
    out["consensus_support"] = support
    out["agreement_within_tolerance"] = within
    out["consensus_within_tolerance"] = within_label
    return out


@dataclass
class BackendComparison:
    """Output of :func:`compare_backends`."""

    backends: list[str]
    cell_agreement: float
    pairwise: pd.DataFrame
    state_concordance: pd.DataFrame
    status_table: pd.DataFrame | None
    labels: pd.DataFrame

    @property
    def status_agreement(self) -> float:
        """Fraction of states whose status is identical across the backends (``nan`` without routes)."""
        if self.status_table is None or not len(self.status_table):
            return float("nan")
        return float(self.status_table["status_agreement"].mean())

    @property
    def status_agreement_within_tolerance(self) -> float:
        """Fraction of states with a label consistent with every backend within its reported margin."""
        if self.status_table is None or "agreement_within_tolerance" not in self.status_table:
            return float("nan")
        return float(self.status_table["agreement_within_tolerance"].mean())

    def summary(self) -> str:
        """
        A short text summary: the backends, all-backend cell agreement, per-state concordance
        and, when transport routes were compared, the status agreement.

        Returns
        -------
        str
        """
        lines = [
            f"backends: {', '.join(self.backends)}",
            f"  all-backend cell agreement {self.cell_agreement:.3f}",
            "  per-state concordance: "
            + ", ".join(f"{s}={v:.2f}" for s, v in self.state_concordance["concordance"].items()),
        ]
        if self.status_table is not None:
            lines.append(
                f"  transport status agreement across backends {self.status_agreement:.3f}"
                + (
                    f" (within the reported margin {self.status_agreement_within_tolerance:.3f})"
                    if np.isfinite(self.status_agreement_within_tolerance)
                    else ""
                )
            )
            cons = self.status_table["consensus"].value_counts()
            lines.append(
                "  consensus status: "
                + ", ".join(f"{k}={int(v)}" for k, v in cons.items())
                + f"; median consensus support {np.nanmedian(self.status_table['consensus_support']):.2f}"
            )
        return "\n".join(lines)


def compare_backends(
    results: dict[str, Predictions | TransportRoute],
    *,
    states: list[str] | None = None,
    group_col: str | None = None,
    tolerance: str = "cutoff",
) -> BackendComparison:
    """
    Compare labels (and statuses) from several backends on the same cells.

    Parameters
    ----------
    results
        ``{backend name: Predictions or TransportRoute}``; all must cover the same cells.
    states
        States to report; defaults to the union of predicted labels.
    tolerance
        Which reported margin the within-tolerance agreement reads when routes are given:
        ``"cutoff"`` (each backend's ``status_alternative``, the fixed ``shift_tolerance``),
        ``"interval"`` (``status_interval_alternative``, the donor-level interval) or
        ``"none"`` (exact agreement only). See :func:`consensus_status`.
    """
    if tolerance not in ("cutoff", "interval", "none"):
        raise ValueError("tolerance must be 'cutoff', 'interval' or 'none'")
    if len(results) < 2:
        raise ValueError("compare_backends needs at least two backends")
    names = list(results)
    labels = {}
    for name, r in results.items():
        if isinstance(r, TransportRoute):
            labels[name] = r.cells["state"].astype(str)
        else:
            labels[name] = r.labels.astype(str)
    L = pd.DataFrame(labels)
    if L.isna().any().any():
        raise ValueError("backends do not cover the same cells")
    all_agree = L.nunique(axis=1) == 1
    pairwise = pd.DataFrame(1.0, index=names, columns=names)
    for a, b in combinations(names, 2):
        v = float((L[a] == L[b]).mean())
        pairwise.loc[a, b] = pairwise.loc[b, a] = v
    if states is None:
        states = sorted(set(np.unique(L.to_numpy())) - {"unknown"})
    rows = []
    for s in states:
        any_s = (L == s).any(axis=1)
        all_s = (L == s).all(axis=1)
        rows.append(
            {
                "state": s,
                "n_cells_any": int(any_s.sum()),
                "n_cells_all": int(all_s.sum()),
                "concordance": float(all_s.sum() / any_s.sum()) if any_s.any() else np.nan,
            }
        )
    conc = pd.DataFrame(rows).set_index("state")
    status = None
    routes = {n: r for n, r in results.items() if isinstance(r, TransportRoute)}
    if len(routes) == len(results):
        status = pd.DataFrame({n: r.states["status"] for n, r in routes.items()}).reindex(states)
        col = {"cutoff": "status_alternative", "interval": "status_interval_alternative"}.get(tolerance)
        alternative = None
        if col is not None:
            alternative = pd.DataFrame(
                {n: (r.states[col] if col in r.states else pd.Series("", index=r.states.index)) for n, r in routes.items()}
            ).reindex(states)
        status = consensus_status(status, alternative)
    return BackendComparison(
        backends=names,
        cell_agreement=float(all_agree.mean()),
        pairwise=pairwise,
        state_concordance=conc,
        status_table=status,
        labels=L,
    )
