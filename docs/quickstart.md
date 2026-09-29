# Quickstart

Everything below runs on simulated cohorts and is deterministic given the seed.

```python
import statetrace as st
from statetrace.datasets import simulate_transport

ref_adata, query_adata, truth = simulate_transport(
    n_ref_groups=8, n_query_groups=6, n_states=5, n_latent=10,
    shifted_states={"S1": 9.0}, absent_states=["S4"], minority_states={"S3": 0.25}, seed=5,
)
```

## 1. Build and validate the reference by donor

```python
ref = st.Reference(ref_adata, group_key="donor_id", state_key="cell_state", latent_key="X_latent").fit()
res = ref.leave_one_group_out()          # one fold per donor
print(res.summary())
res.state_metrics[["n_groups_reference", "median_group_recall", "group_recovery_rate", "reproducibility"]]
ctrl = ref.random_split(n_splits=5)      # the conventional, optimistic control
```

`leave_one_study_out()` does the same over a `study_key`. To validate an external mapper
instead of the built-in kNN, wrap it in {class}`~statetrace.CallableBackend` or run it in your
own leave-one-donor-out loop and summarise the held-out table with
{func}`~statetrace.summarize_holdout`.

## 2. Freeze

```python
path = ref.freeze("my_reference.statetrace")
ref = st.Reference.load(path).fit()
```

## 3. Validate transport into an independent cohort

```python
route = st.validate_transport(ref, query_adata, group_key="donor_id")
print(route.summary())
route.states[["ref_reproducibility", "groups_recovered", "query_groups", "shift_level", "ood_fraction", "status"]]
```

With predictions from another tool:

```python
preds = st.Predictions.from_scpoli(query_adata)                 # or from_scanvi / from_symphony / from_celltypist / from_obs
route_scpoli = st.validate_transport(ref, query_adata, group_key="donor_id", predictions=preds)
st.compare_backends({"knn": route, "scpoli": route_scpoli}).summary()
```

## 4. Gate downstream inference

```python
route.supported_states
abundance = route.state_abundance(states="supported")          # donor x state
pb = route.pseudobulk(states="supported")                      # donor x state pseudobulk AnnData
route.report("results/statetrace_report")                      # tables + report.md
route.to_obs()                                                 # writes statetrace_* columns into query.obs
```
