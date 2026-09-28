# The frozen reference format

{meth}`~atlasroute.AtlasReference.freeze` writes a directory named `<name>.atlasroute`
(or `<name>.atlasroute.zip` when the path ends in `.zip`). Nothing is pickled; the
bundle can be read with numpy, pandas and json alone.

```text
name.atlasroute/
├── manifest.json     format_version, atlasroute and library versions, created, reference_id,
│                     keys {donor, state, celltype, latent}, latent provenance, backend {name, params},
│                     counts, thresholds, validated, files {path: sha256}
├── latent.npy        float32 (n_cells, n_dims)
├── cells.npz         obs_names, state_code, donor_code, celltype_code (-1 when absent)
├── categories.json   {"states": [...], "donors": [...], "celltypes": [...] | null}
├── centroids.npy     float32 (n_states, n_dims) state means in latent space
├── state_table.csv   n_cells, n_donors, eligible, reason per state
└── validation/       present when leave_one_donor_out() ran before freeze()
    ├── config.json   method, LODOConfig, backend, latent provenance, versions
    ├── overall.json
    ├── state_metrics.csv
    ├── donor_metrics.csv
    ├── donor_summary.csv
    ├── confusion_matrix.csv
    ├── cell_predictions.csv.gz   (optional; include_cell_predictions=True)
    └── per_donor_confusion/      (optional)
```

{func}`~atlasroute.read_frozen` verifies `format_version` against what the installed
AtlasRoute supports and recomputes the SHA-256 of every listed file, raising
{class}`~atlasroute.FrozenReferenceError` on any mismatch. The k-nearest-neighbour index
is never stored: {meth}`~atlasroute.AtlasReference.load` followed by
{meth}`~atlasroute.AtlasReference.fit` rebuilds it from `latent.npy` in seconds, which
keeps the format independent of scikit-learn internals.

Tables are CSV rather than parquet so the format adds no dependency; this may change if
cell-level tables become large.
