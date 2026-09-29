"""
The standardised transport report: the evidence, per state, that a state defined in one cohort
is the same construct analysed in another. Written as CSV tables plus a Markdown summary a
reader (or reviewer) can open without StateTrace.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from . import schema

if TYPE_CHECKING:
    from .transport import TransportRoute


def _md_table(df: pd.DataFrame, *, index_name: str = "state", floatfmt: str = ".2f") -> str:
    cols = [index_name, *df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else format(v, floatfmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join([str(idx), *vals]) + " |")
    return "\n".join(lines)


def write_report(
    route: TransportRoute, output_dir: str | Path, *, include_cells: bool = True
) -> Path:
    """Write ``states.csv``, ``groups.csv``, ``cells.csv.gz``, ``config.json`` and ``report.md``."""
    from .transport import route_to_dir

    out = route_to_dir(route, output_dir, include_cells=include_cells)
    st = route.states
    counts = st["status"].value_counts().reindex(schema.TRANSPORT_STATUS, fill_value=0)
    show = st[
        [
            "ref_groups",
            "ref_reproducibility",
            "groups_recovered",
            "query_groups",
            "recovery_rate",
            "median_confidence",
            "uncertain_fraction",
            "ood_fraction",
            "shift_level",
            "status",
        ]
    ].copy()
    show["groups_recovered"] = (
        show["groups_recovered"].astype(str) + "/" + show["query_groups"].astype(str)
    )
    show = show.drop(columns="query_groups")
    md = [
        "# StateTrace transport report",
        "",
        f"Backend: **{route.predictions.backend}** | query groups (`{route.group_key}`): {route.cells['group'].nunique()} | "
        f"query cells: {len(route.cells)} | reference states: {len(st)} | created {route.created} | StateTrace {route.statetrace_version}",
        "",
        "## Transport status",
        "",
        ", ".join(f"**{k}** {int(v)}" for k, v in counts.items()),
        "",
        "States marked `supported` have enough group-level evidence to enter downstream "
        "differential-abundance, pseudobulk or genetic analyses; `supported_with_caution` states "
        "should be reported with their caveat; `shifted`, `insufficient` and `absent` states should not "
        "be analysed as the reference construct.",
        "",
        "## Evidence per state",
        "",
        _md_table(show),
        "",
        "Columns: reference groups supporting the state and its grouped-holdout reproducibility; "
        "query groups that recovered the state (of all query groups); recovery rate among groups "
        "observing it; median mapping confidence; fraction of its cells that are uncertain or "
        "out of reference; centroid shift in reference-spread units.",
        "",
        "## Rules",
        "",
        "```",
        *(f"{k} = {v}" for k, v in route.config.to_dict().items()),
        "```",
        "",
        "Reference: "
        + ", ".join(f"{k}={v}" for k, v in route.reference.items() if k != "latent")
        + f"; latent={route.reference.get('latent', {}).get('source')}",
    ]
    (out / "report.md").write_text("\n".join(md) + "\n")
    return out
