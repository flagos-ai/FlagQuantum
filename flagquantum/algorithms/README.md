# Algorithms

Algorithms assemble user-facing quantum and hybrid workflows from FlagQuantum
circuits, observables, differentiation, and optimization. This package owns
Hamiltonian helpers, reusable ansatz construction, VQE/QAOA-style workflows,
Trotterized time evolution, and staged optimization recipes whose scientific
behavior is tested end to end.

Algorithms do not define circuit or operator semantics, compiler passes,
runtime selection, provider lifecycle, numerical kernels, or benchmark claims.
New workflows should compose supported FlagQuantum APIs and must not introduce
external-framework objects or a second execution path.

`docs/guides/ALGORITHMS.md` is the per-unit reference for the units it indexes:
what each one does, where its construction comes from, and the advantage premise
it rests on. Runnable demonstrations live under
[`examples/algorithms/`](../../examples/algorithms/README.md), and each one is
executed by `tests/test_algorithm_examples.py`.

## Where to start

- `core.py`: Hamiltonians, ansatz builders, losses, and complete algorithm
  workflows.
- `error_mitigation.py`: zero-noise extrapolation — one observable measured at
  several error strengths by scaling the single error-probability parameter each
  noise channel declares, then continued to zero by polynomial least squares or
  Richardson extrapolation, with the residual it left and the variance
  amplification of its weights reported beside the estimate. A channel whose
  parameters are not error probabilities is refused by name, and a model that
  declares a readout rule is refused rather than measured without it, because the
  estimate is `Tr(O rho)` and classical readout confusion is applied after
  measurement. Clifford data regression and readout-error mitigation are absent,
  and probabilistic error cancellation is a separate unit beside it.
- `pec.py`: probabilistic error cancellation — each declared Pauli channel is
  inverted from its Pauli transfer matrix into an exact signed combination of
  Pauli words, which is inserted after the channel it inverts and summed
  exactly. A channel whose transfer matrix is not diagonal, and one whose
  transfer spectrum reaches zero, are refused by name; `gamma` and the `gamma**2`
  shot cost a sampled implementation would pay are reported beside the estimate,
  together with the term count the run actually spends. Clifford data regression
  and readout-error mitigation are absent.
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
  These take a PyTorch optimizer or one of the staged methods below; `spsa.py` is
  the gradient-free member of the same surface and takes a plain callable instead.
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
- `svd.py`: singular values read from the phase of a matrix's Hermitian
  embedding, with a private block encoding of that embedding beside the readout.
  Demonstration scale: the matrix, its embedding, the embedding's exponential and
  the input state are all formed classically, the input state from the matrix's
  own singular vectors.
- `spsa.py`: simultaneous perturbation stochastic approximation — a gradient-free
  optimizer for objectives whose only accessible value is a sample, at two
  evaluations per step whatever the parameter count. **The estimate is biased for
  every finite perturbation and is not a gradient**: an objective with an exact
  gradient is served more cheaply and exactly by autograd or parameter shift, and
  the reason to use this unit is its evaluation cost. Demonstration scale: the
  perturbation is drawn from a caller-owned `torch.Generator` so a run replays.
- `trotter.py`: time evolution by a product formula — a weighted Pauli sum
  becomes an ordinary `Circuit` whose unitary approximates `exp(-i * time * H)`,
  with the exact circuit for one Pauli word's exponential underneath it, as a
  basis change, a CX ladder, and one rotation. Two orders are built, 1 and 2, and
  neither comes with a bound: the product is exact only when the terms commute,
  and how far it is from exact otherwise is a number the caller measures, because
  a bound needs a commutator norm the caller has. An identity term is refused
  rather than dropped, since no gate here applies a global phase, and the declared
  term order is preserved rather than sorted. There is no `exp_pauli` opcode:
  emitting the decomposition is what lets the compiler's own passes count, route,
  and differentiate it. Demonstration scale: the exact side of every comparison is
  a dense matrix exponential, which exists at the sizes this repository simulates.
- `logical_resources.py`: logical-layer resource estimation — a Clifford+T
  program's operation counts, T family, and schedule depth, read out of
  `compiler/resource_estimation.py` and costed on a rotated surface code at a
  distance the caller names: `2 d^2 - 1` physical qubits per patch and `d`
  surface-code cycles per logical layer. **A count and a footprint, and not a
  measurement or a failure rate**: nothing runs, no wall-clock time or memory is
  read, and no logical error rate is reported, because a threshold fit's
  prefactor and threshold belong to the device rather than to this unit. The
  input has to already be Clifford+T, so a parametric rotation is refused
  rather than synthesised, and a compound operation whose T-count is its
  decomposition's — a Toffoli, a controlled swap — is refused because a Toffoli
  is not a Clifford gate and counting one as a single Clifford operation would
  understate the T-count. No distillation factory, magic-state budget, routing
  overhead, placement, scheduling, or device model is included, so the physical
  figure is a floor for a circuit of these layers rather than a compiled
  estimate.
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
