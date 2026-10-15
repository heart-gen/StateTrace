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


@dataclass
class BackendComparison:
    """Output of :func:`compare_backends`."""

    backends: list[str]
    cell_agreement: float
    pairwise: pd.DataFrame
    state_concordance: pd.DataFrame
    status_table: pd.DataFrame | None
    labels: pd.DataFrame

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
            agree = float(self.status_table["status_agreement"].mean())
            lines.append(f"  transport status agreement across backends {agree:.3f}")
        return "\n".join(lines)


def compare_backends(
    results: dict[str, Predictions | TransportRoute],
    *,
    states: list[str] | None = None,
    group_col: str | None = None,
) -> BackendComparison:
    """
    Compare labels (and statuses) from several backends on the same cells.

    Parameters
    ----------
    results
        ``{backend name: Predictions or TransportRoute}``; all must cover the same cells.
    states
        States to report; defaults to the union of predicted labels.
    """
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
        status["status_agreement"] = status.nunique(axis=1) == 1
        status["consensus"] = [
            row.drop("status_agreement").mode().iloc[0]
            if row.drop("status_agreement").notna().any()
            else None
            for _, row in status.iterrows()
        ]
    return BackendComparison(
        backends=names,
        cell_agreement=float(all_agree.mean()),
        pairwise=pairwise,
        state_concordance=conc,
        status_table=status,
        labels=L,
    )
