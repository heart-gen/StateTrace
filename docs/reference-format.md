# The frozen reference format

{meth}`~statetrace.StateReference.freeze` writes a directory named `<name>.statetrace`
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
├── geometry.npy      float32 (n_states, 3) RMS radius, mean and sd of cell-to-centroid distance per state
├── state_table.csv   n_cells, n_groups, n_studies, eligible, reason per state
├── expression/       the frozen biological space (format 3; absent when the reference had no genes)
│   ├── params.json   ExpressionConfig plus input kind (counts / non_count) and the input gene count
│   ├── genes.json    the selected genes, in the column order of every array below
│   ├── mu.npy        float32 (n_states, n_genes) reference donor-mean pseudobulk profile per state
│   ├── sigma.npy     float32 (n_states, n_genes) shrunk between-donor sd per state
│   ├── n_groups.npy  int32 (n_states,) reference donors behind each state
│   ├── components.npy float32 (n_components, n_genes) PCA loadings on the standardised reference rows
│   ├── profiles.npy  float32 (n_rows, n_genes) reference donor x state log-normalised pseudobulks
│   ├── rows.csv      group, state (codes), n_cells for each profile row
│   ├── null.csv      held-out reference donor metrics (distance, distance_pc, residual, nearest, genes_beyond)
│   └── null_stats.csv n_holdout, mean, sd, q95, q99 of the held-out distance per state
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
StateTrace supports (format 2 bundles, which predate `expression/`, still load and simply
carry no expression space) and recomputes the SHA-256 of every listed file, raising
{class}`~statetrace.FrozenReferenceError` on any mismatch. The k-nearest-neighbour index
is never stored: {meth}`~statetrace.StateReference.load` followed by
{meth}`~statetrace.StateReference.fit` rebuilds it from `latent.npy` in seconds, which
keeps the format independent of scikit-learn internals.

Tables are CSV rather than parquet so the format adds no dependency; this may change if
cell-level tables become large.
