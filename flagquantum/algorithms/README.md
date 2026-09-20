# Algorithms

Algorithms assemble user-facing quantum and hybrid workflows from FlagQuantum
circuits, observables, differentiation, and optimization. This package owns
Hamiltonian helpers, reusable ansatz construction, VQE/QAOA-style workflows,
and staged optimization recipes whose scientific behavior is tested end to
end.

Algorithms do not define circuit or operator semantics, compiler passes,
runtime selection, provider lifecycle, numerical kernels, or benchmark claims.
New workflows should compose supported FlagQuantum APIs and must not introduce
external-framework objects or a second execution path.

## Where to start

- `core.py`: Hamiltonians, ansatz builders, losses, and complete algorithm
  workflows.
- `feature_selection.py`: feature selection as a QUBO — a subset's relevance and
  redundancy scored with a penalty on the size of the subset, built for a solver
  and evaluated at an assignment. No annealer is supplied: the repository has
  none, and any advantage a solver observes is the solver's.
- `kmedians.py`: Grover-quantized k-medians — each point's nearest centroid is
  found by a Grover-style minimum search over a centroid index register, and the
  centroid positions are then moved to classical coordinate-wise medians.
  Demonstration scale: the distance table is classical and the search is bound
  at three register wires.
- `optimization.py`: reusable classical and quantum-aware optimization stages.
- `pca.py`: quantum PCA — the eigenvalue readout of a data matrix's density
  matrix, by phase estimation over its exponential. Demonstration scale: the
  density matrix and its exponential are formed classically.
- `qarm.py`: the frequent-itemset fraction by amplitude estimation — the uniform
  superposition over the items, a support register the circuit fills one controlled
  increment per transaction-item membership, and a mark at the support threshold.
  Demonstration scale: the transactions are iterated classically, in place of the coherent
  database access the cited paper assumes and this unit does not exercise.
- `quantum_kernel.py`: kernel entries by swap test, with a classical kernel ridge
  classifier over them. Demonstration scale: the feature map is this package's
  own angle encoding, built gate by gate on every run, and every entry is a
  sample rather than a reading.
- `__init__.py`: the intentionally small public algorithms surface.
- `primitives/`: shared quantum primitives. Its contents are admitted only when at least two
  algorithm modules need them.

Staged optimization owns contiguous copies of input parameter groups. This
lets Rotosolve update coordinates in place and LBFGS flatten gradients even
when callers supply transposed tensors. Input values, shapes, precision, and
device placement are preserved; caller-owned tensors are never optimized in
place. The parameter-layout tests in `tests/test_hybrid_optimization.py` cover
both methods with contiguous and transposed inputs.

`core.py` accepts existing Circuit, MPS, statevector, and density-matrix inputs,
but dense Pauli-product mathematics lives in `simulation/pauli.py`. Keep new
numerical kernels in Simulation and preserve Algorithms as their workflow
composition layer.

## Ten-minute change path

For a small algorithm change, modify one owning module, add a scenario-level
test that demonstrates convergence or the expected failure, and run:

```bash
python -m pytest tests/unit/test_algorithms_package.py \
  tests/test_hybrid_optimization.py \
  tests/integration/test_cpu_vertical_slice.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Keep defaults deterministic, preserve requested precision and autograd, and
avoid embedding device selection or performance assertions in an algorithm.
Changes to root exports or protected result types require the public API change
process rather than an algorithms-local compatibility wrapper.
