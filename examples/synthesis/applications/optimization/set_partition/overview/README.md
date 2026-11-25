# Native set-partition Notebook

From the repository root:

```bash
python -m examples.synthesis.applications.optimization.set_partition.overview.main
```

Open `walkthrough.ipynb` for source-seeded data, a parameterized native quantum
function, synthesis budgets, actual optimization, 1000 optimized and 1000 uniform
samples, cost plots and the observed partition. No foreign quantum code is loaded.

The source Notebook data is `(6, 1, 4, 4, 8, 10, 4, 6, 3, 5)` (total 51).
The separate frozen Qmod counterpart uses different data and remains unchanged.
Ten decision bits minimize the squared difference of the group sums, using three
phase/RX layers, CVaR quantile 0.7 and an 80-iteration budget.

Public Nelder-Mead replaces the source optimizer; exact local probabilities
replace its shot estimator during search. Final observations are actual seeded
CPU samples. Report convergence separately from seeing a good sampled partition.
A bounded exact reference is computed after sampling and never supplies answers.
See [the source-cell mapping](WORKFLOW.md) for retained semantics and replacements.
