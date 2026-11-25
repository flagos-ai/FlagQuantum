# Complete discrete Poisson workflow

```sh
python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.overview.main
```

Open `walkthrough.ipynb` for all 16 original stages. The complete 8 by 8 grid,
six estimator bits and 13 physical qubits are retained. Fresh source generation,
Hamiltonian construction, local execution, postselection, classical solve and
both contour figures run without source importers.

Complete complex state validation uses an independent dense Circuit. Source-scale
classical error is explicitly reported, not treated as numerical success. The
source integer reciprocal table is retained while the native QPE register exposes
normalized fractional phases; zero postselection is unchanged. Static transform
views use explicit unitary matrices within the current public contract.

Outputs: `artifacts/development/discrete_poisson/`. The frozen quantum-program
counterpart is unchanged. See `provenance.json` for reviewed source stages.
