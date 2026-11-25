# Native maximum k-vertex cover Notebook

Run the complete local workflow and independent Circuit comparison from the
repository root:

```bash
python -m examples.synthesis.applications.optimization.max_k_vertex_cover.overview.main
python -m examples.synthesis.applications.optimization.max_k_vertex_cover.comparisons.circuit
```

Open `walkthrough.ipynb` for the original ten-node graph (edge probability 0.5,
seed 13), K=5, native three-layer quantum functions, explicit synthesis budgets,
actual CVaR optimization, 1000 optimized and 1000 uniform samples, and graph plots.
The native source files and Notebook do not load foreign quantum sources.

Minimize uncovered edges subject to selecting exactly five nodes. The quantum
cost adds ten times the squared cardinality violation. Keep the original
90-iteration setting and CVaR quantile 0.7; report feasibility separately from
penalized cost. No feasible observation remains `None`, without substituting the
exact classical reference. Array item i is bit i of the named v integer.

Public native Nelder-Mead and exact local CVaR replace the source private
optimizer and shot estimation during search. Final observations are actual
seeded CPU samples. A separately labeled bounded reference replaces Couenne only
after sampling. A sampled optimum does not prove convergence or quantum advantage.
[Source-cell mapping and limits](WORKFLOW.md) documents every replacement.
