# Hierarchical QAOA for Max-Cut

Both source code stages, the fresh 40-node graph with edge probability 0.2, a six-qubit cap per community, two QAOA layers, and 20 objective queries per community. Signed contractions and original-node flips, finite sample ranking, both classical comparisons and the complete labeled graph. Independent absolute Circuit states and cut tables for all realized quantum subproblems.

Each community uses a native Python quantum function that phases the weighted cut expression and mixes the decision bits. The classical workflow contracts the graph using signed weights and propagates the resulting flips back to the original 40 nodes.

From the repository root:

```bash
python -m examples.synthesis.applications.optimization.qaoa_in_qaoa.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) to inspect each subproblem and view the resulting cut. This example uses NetworkX, NumPy, and Matplotlib in the example environment. [Provenance](provenance.json) records the source and migration boundaries.

Native public COBYLA with a hard 20-query cap uses the exact mean negative cut. The initial schedule follows the norm formula reviewed in current Classiq SDK 1.35.0. The source leaves the historical SDK and shot default unspecified; use 1000 actual CPU samples per quantum subproblem, seed 10, and resolve identical cut/count ties by ascending word. Every trial specializes a fresh native program at synthesis. Keep Python seed 246 and use NumPy seed 4812 for layout with local RNGs. No frozen subproblem, historical trajectory equivalence, runtime parameter binding, hardware or optimum claim.
