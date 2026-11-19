# Three-asset portfolio optimization

The complete native workflow preserves the original covariance matrix, return
vector, integer holdings in 0..7, budget 6, three QAOA layers and penalty 10.
Users express the cost and constraints as Python computations, bind classical
parameters to the native quantum function and read holdings by register name.

Run from the repository root:

```sh
python -m examples.synthesis.applications.finance.portfolio_optimization.overview.main
```

The Notebook runs a bounded native COBYLA search using the lower 70 percent of
the actual encoded-cost distribution, then draws 1,000 optimized observations
and 1,000 uniform observations. It plots the cost trace and weighted cost
histograms, reports feasibility and selects the best feasible observed holding.
Independently enumerate all 512 classical holdings to establish the classical
optimum. The quantum result is never filled in from the classical optimum.

Exact local probabilities replace the hosted optimizer's estimator; initial
parameters, trust radii, shot counts and seeds are explicit. No claim equates
the search trajectory with a proprietary optimizer or proves quantum advantage.
The parent quantum-only program remains unchanged.

See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json) for the
original-cell mapping, adaptations and acceptance scope.
