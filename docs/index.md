# StateTrace

**A backend-independent framework for donor- and cohort-aware validation of cellular-state
transportability and downstream inference.**

StateTrace does not map cells. Mapping tools (scPoli, scANVI/scArches, Symphony, Azimuth,
CellTypist, a user classifier, or StateTrace's own kNN baseline) answer *which state is this
cell?* StateTrace answers a different question: **is a cellular state defined in one human
cohort reproducible and transportable enough to support inference in another?**

For every state it asks, in order: is it reproducible across the reference's donors, or does
it depend on particular donors? Does it recover in unseen donors, and in an independent
cohort? Is the conclusion the same across mapping backends? Is the query version shifted,
uncertain or partly out of reference? Only then is the state **supported** for group-level
differential abundance, pseudobulk expression or genetic analyses.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/01_validate_reference
tutorials/02_freeze_and_reload
tutorials/03_transport_query
tutorials/04_two_h5ads_to_report
```

```{toctree}
:maxdepth: 2
:caption: Guides

validation
transport
reference-format
roadmap
staged-roadmap
spatial-roadmap
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
