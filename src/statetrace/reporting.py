"""
The standardised transport report: the evidence, per state, that a state defined in one cohort
is the same construct analysed in another. Written as CSV tables plus a Markdown summary a
reader (or reviewer) can open without StateTrace.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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


def _two_sample_section(route: TransportRoute) -> list[str]:
    """Markdown for the two-sample engines (empty when they did not run)."""
    t = route.two_sample
    if t is None or len(t) == 0:
        return []
    ok = t[(t["status"] == "ok") & (t["subset"] == "all")]
    wide = (
        ok.assign(column=ok["engine"] + " " + ok["level"] + " p")
        .pivot_table(index="state", columns="column", values="p_value", aggfunc="first")
        .reindex(route.states.index)
    )
    not_ok = t.loc[t["status"] != "ok", ["engine", "level", "status"]].value_counts()
    lines = [
        "## Two-sample engines (reported, not decisive)",
        "",
        "Whether each reference state and the query cells assigned to it come from the same "
        "distribution, by E-distance and/or scXMatch, at cell level (sampled cells in the "
        "frozen cell space) and at donor level (donor x state pseudobulk profiles). These "
        "p-values are evidence of *a* difference; they do not decide `status`, and a cell-level "
        "p-value treats cells of one donor as independent. Full table: `two_sample.csv`.",
        "",
        _md_table(wide, floatfmt=".3g"),
        "",
    ]
    if len(not_ok):
        lines += [
            "Rows not run: "
            + "; ".join(f"{e}/{lv}: {st} x{n}" for (e, lv, st), n in not_ok.items())
            + ".",
            "",
        ]
    return lines


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
            "shift_source",
            "groups_shifted",
            "novel_population",
            "status",
            "caution_reasons",
        ]
    ].copy()
    show["groups_recovered"] = (
        show["groups_recovered"].astype(str) + "/" + show["query_groups"].astype(str)
    )
    show["groups_shifted"] = (
        show["groups_shifted"].astype(str) + "/" + st["groups_scored_expression"].astype(str)
    )
    show = show.drop(columns="query_groups")
    show.insert(
        show.columns.get_loc("shift_level"),
        "shift_raw/residual",
        [
            f"{raw:.1f}/{res:.1f}" if np.isfinite(raw) and np.isfinite(res) else ""
            for raw, res in zip(
                np.where(
                    st["shift_source"] == "expression",
                    st["expression_shift_raw"],
                    st["latent_shift_raw"],
                ),
                st["shift"],
                strict=True,
            )
        ],
    )
    offsets = (
        route.groups.groupby("group")[["latent_offset_sd", "expression_offset_rms"]]
        .first()
        .rename(
            columns={
                "latent_offset_sd": "latent offset (sd)",
                "expression_offset_rms": "expression offset (rms z)",
            }
        )
    )
    expr = route.reference.get("expression", {}) or {}
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
        "out of reference; the shift before and after donor-offset removal (raw/residual, in "
        "reference sd units; the residual decides the level); shift level and the coordinate "
        "system it was judged in (`expression`: "
        "median over query donors of the donor x state pseudobulk distance, standardised against "
        "held-out reference donors in the frozen expression space; `latent`: centroid shift in "
        "per-dimension reference sd in the mapping latent; both after cohort-offset removal); "
        "query donors whose pseudobulk lies at or beyond the high-shift threshold (of donors "
        "scored); whether an out-of-reference population sits next to the state.",
        "",
        "## Coordinate systems",
        "",
        "State identity (label, confidence, per-cell reference distance) comes from the mapping "
        f"space of **{route.predictions.backend}**"
        + (
            " (query latent available)"
            if route.reference.get("latent_available")
            else " (no query latent: per-cell distance unavailable)"
        )
        + (
            f". Local reference support: {route.reference.get('support', {}).get('source')}; "
            f"confidence source: {route.reference.get('confidence_source')}; "
            f"out-of-reference rule: {route.reference.get('ood_rule')}"
        )
        + ". Per-donor offsets removed before the residual shift (latent, in per-dimension "
        "reference sd; expression, RMS of the standardised profile):"
        + "\n\n"
        + _md_table(offsets, index_name="donor")
        + "\n\nState shift comes from the frozen expression space"
        + (
            f" ({expr.get('n_genes_scored')} genes, feature retention "
            f"{expr.get('feature_retention', float('nan')):.0%}, cohort offset RMS "
            f"{expr.get('cohort_offset_rms') if expr.get('cohort_offset_rms') is not None else float('nan'):.2f} sd)."
            if expr.get("available")
            else " when the reference carries one; this reference does not, so shift falls back to the mapping latent."
        ),
        "",
        *_two_sample_section(route),
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
