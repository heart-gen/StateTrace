# AtlasRoute

**Donor-aware building, freezing, validation and transport of cellular-state references
across single-cell and single-nucleus datasets.**

AtlasRoute answers one question: *we discovered this cell state in dataset A; can we
legitimately carry it into dataset B?* It treats donors, not cells, as the replication
unit; separates mapping from validation; allows a reference to abstain (`unknown`); and
freezes a reference before any query is examined.

Version 0.1 delivers the methodological core: leave-one-donor-out validation of a
labelled reference in a fixed latent space, donor x state metrics, and a frozen
reference format. Query transport, calibration and novelty detection follow in 0.2.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 2
:caption: Guides

validation
reference-format
roadmap
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
