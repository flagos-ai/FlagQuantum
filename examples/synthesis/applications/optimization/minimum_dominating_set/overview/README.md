# Minimum dominating set

Find a smallest subset of nodes that covers every node by membership or adjacency.

All fifteen code stages: fresh six-node graph (edge probability 0.6, seed 8), six decision bits, fourteen encoded slack bits, six QAOA layers, penalty 10, 70 optimizer iterations, 70% lower-tail CVaR, optimized and uniform samples, and an independent exhaustive constrained solution.

From the repository root:

```bash
python -m examples.synthesis.applications.optimization.minimum_dominating_set.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for the original graph, sampled cost distribution, and side-by-side observed and exact sets. The filter used in the independent search is not applied to repair the samples.

Reuse the native qfunc for the original seed-8 closed neighborhoods and retain every slack register. Exact local CVaR integrates all joint decision/slack masses. Native Nelder–Mead and an explicit linear pi schedule replace the unpinned hosted optimizer; actual query counts are shown separately from the iteration budget. Use 1000 actual CPU draws for each comparison. NetworkX seeded spring layout renders the same underlying graph with selected nodes and incident edges; it does not reproduce hosted layout coordinates. Exhaustive search over 64 decision assignments replaces Couenne and only runs after sampling. No historical optimizer trajectory or optimum guarantee for the observed quantum sample.

See [provenance.json](provenance.json) for the upstream and code-stage pins.
