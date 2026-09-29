# The frozen reference format

{meth}`~statetrace.AtlasReference.freeze` writes a directory named `<name>.statetrace`
(or `<name>.statetrace.zip` when the path ends in `.zip`). Nothing is pickled; the
bundle can be read with numpy, pandas and json alone.

```text
name.statetrace/
├── manifest.json     format_version, statetrace and library versions, created, reference_id,
│                     keys {group, state, celltype, study, latent}, latent provenance, backend {name, params},
│                     counts, thresholds, validated, files {path: sha256}
├── latent.npy        float32 (n_cells, n_dims)
├── cells.npz         obs_names, state_code, group_code, study_code, celltype_code (-1 when absent)
├── categories.json   {"states": [...], "groups": [...], "studies": [...] | null, "celltypes": [...] | null}
├── centroids.npy     float32 (n_states, n_dims) state means in latent space
├── spread.npy        float32 (n_states,) RMS radius per state (novelty unit)
├── state_table.csv   n_cells, n_groups, n_studies, eligible, reason per state
└── validation/       present when leave_one_group_out() ran before freeze()
    ├── config.json   method, LODOConfig, backend, latent provenance, versions
    ├── overall.json
    ├── state_metrics.csv
    ├── group_metrics.csv
    ├── group_summary.csv
    ├── confusion_matrix.csv
    ├── cell_predictions.csv.gz   (optional; include_cell_predictions=True)
    └── per_group_confusion/      (optional)
```

{func}`~statetrace.read_frozen` verifies `format_version` against what the installed
StateTrace supports and recomputes the SHA-256 of every listed file, raising
{class}`~statetrace.FrozenReferenceError` on any mismatch. The k-nearest-neighbour index
is never stored: {meth}`~statetrace.AtlasReference.load` followed by
{meth}`~statetrace.AtlasReference.fit` rebuilds it from `latent.npy` in seconds, which
keeps the format independent of scikit-learn internals.

Tables are CSV rather than parquet so the format adds no dependency; this may change if
cell-level tables become large.
