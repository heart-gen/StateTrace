# Quickstart

The whole v0.1 workflow on a simulated reference. Every call is deterministic given the seed.

```python
import atlasroute as ar
from atlasroute.datasets import simulate_reference

adata = simulate_reference(n_donors=8, n_states=5, n_cells_per_donor=200, donor_effect=0.7, seed=1)

ref = ar.Reference(adata, donor_key="donor_id", state_key="cell_state", latent_key="X_latent").fit()
```

`fit()` validates the metadata (donor and state columns become categorical; states with
few donors or cells are flagged), takes the latent coordinates from `obsm["X_latent"]`
(or computes a PCA when `latent_key=None`) and fits the k-nearest-neighbour backend.

## Leave-one-donor-out

```python
res = ref.leave_one_donor_out()
print(res.summary())
res.state_metrics[["n_donors_reference", "median_donor_recall", "min_donor_recall", "lodo_stability"]]
```

`res.donor_metrics` is the long donor x state table, `res.donor_summary` has one row per
donor, `res.confusion_matrix` pools every held-out cell and `res.cell_predictions` keeps
the per-cell probability, margin and neighbour support.

## The optimistic control

```python
ctrl = ref.random_split(n_splits=5)
ctrl.overall["accuracy"], res.overall["accuracy"]
```

A stratified split over cells ignores donor membership and is the conventional benchmark.
The gap between the two accuracies is the donor effect the reference must survive.

## Freeze and reload

```python
path = ref.freeze("nvu_reference.atlasroute")       # a directory; use "....zip" for an archive
ref2 = ar.Reference.load(path).fit()                 # rebuilds the backend from the stored latent
ref2.state_table()
```

## Write predictions back to the AnnData

```python
res.to_obs(adata)          # adds atlasroute_lodo_state, _probability, _margin, _abstained
```
