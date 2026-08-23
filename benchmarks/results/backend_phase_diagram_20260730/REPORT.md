# SV–MPS–TN backend phase diagram

## Scope

The matrix contains 95 selector decisions across:

- 30--1024 qubits;
- shallow one-dimensional chains;
- nonlocal binary trees;
- 5x6, 6x6, and 8x8 grids;
- full state, sampling, one/few amplitudes, and local observables;
- an 80 GiB per-rank memory budget.

Every record, including candidate memory estimates and blockers, is stored in
`matrix.json`.

## Result

| Output contract | SV | MPS | TN |
|---|---:|---:|---:|
| Full state | 19 | 0 | 0 |
| Samples | 3 | 16 | 0 |
| Single amplitude | 3 | 7 | 9 |
| Four amplitudes | 3 | 7 | 9 |
| Eight local observables | 3 | 7 | 9 |

The main sparse-output boundary is:

- At 30 qubits, SV remains the lower-risk default for all tested structures.
- From 36 through 1024 qubits, shallow local chains select MPS.
- From 36 qubits upward, nonlocal low-treewidth binary trees select TN.
- The 36q 6x6 and 64q 8x8 grids select TN after dense capacity failure.

This matches the saved measurements:

- 40q depth-4 chain: MPS bond 16 measured in the crossover campaign.
- 24q nonlocal tree: TN was 546x faster than MPS steady state.
- 36q 4x9 grid: distributed TN achieved 7.26x speedup on eight A800s with a
  16 GiB rank-local working set.

## Correctness fix

The scan exposed and fixed a full-state contract bug.  When dense output exceeds
memory, MPS cannot satisfy the request merely because its internal
representation is compact: returning the requested full state still requires
`2^n` materialization.  Full-state selection now retains statevector semantics
and fails closed in the dense/sharded capacity preflight instead of silently
choosing MPS.

## Interpretation boundary

This is a structural and capacity phase diagram, not a runtime prediction for
every point.  TN selections still require the real contraction-path and
working-set preflight.  MPS selections remain conditional on the estimated bond
and approximation policy.  The next calibration step is measured boundary
points around 30--40 qubits and gradient-specific phase diagrams.
