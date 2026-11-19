# Native QAOA tutorial

Run both original tutorial sections from the repository root:

```bash
python -m examples.synthesis.algorithms.search_and_optimization.qaoa.overview.main
python -m examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.comparisons.circuit
python -m examples.synthesis.algorithms.search_and_optimization.qaoa.comparisons.circuit
```

Open `walkthrough.ipynb` for native quantum functions, explicit synthesis budgets,
actual optimization, seeded readout and plots. MaxCut retains four layers and
Knapsack retains three. Each keeps its original initial parameter schedule,
60-iteration budget and 1000 final samples. Angles are radians.

The public native Nelder-Mead optimizer replaces source COBYLA; exact local
expected costs replace the source shot estimator during optimization. Final
observations are actual seeded CPU samples. These explicit replacements do not
promise an identical optimizer trajectory. Iteration limits can be exhausted
without convergence. An observed optimum does not prove variational convergence.

[Workflow mapping and limits](WORKFLOW.md) lists every source code cell and the
retained domains, cost functions, graph roles and replacement boundaries.
