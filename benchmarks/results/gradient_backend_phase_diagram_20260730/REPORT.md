# Gradient-specific SV–MPS–TN phase diagram

## Matrix

The matrix contains 36 gradient decisions for complex128 expectation and eight
local observables under an 80 GiB per-rank budget.  It covers 24--256 qubits,
shallow chains, nonlocal binary trees, and 2D grids up to 8x8.

| Output contract | SV | MPS | TN |
|---|---:|---:|---:|
| Expectation gradient | 6 | 5 | 7 |
| Eight-observable gradient | 6 | 5 | 7 |

The boundary is:

- 24--30q: statevector remains the lower-risk exact reverse path.
- 36--256q shallow local chains: MPS.
- 36--256q nonlocal low-treewidth trees: TN.
- 36q 6x6 and 64q 8x8 grids: TN.

## Gradient-specific correction

MPS reverse capacity was previously estimated with forward storage only.
The selector now applies a 3x factor for the primal MPS, left/right
environments, and factorization work buffers.  The runtime checkpoint and
factorization preflights remain authoritative.

Statevector reverse already uses a 3x primal/adjoint/work-buffer estimate.
TN gradient memory increases the structural width proxy and still requires a
real sliced reverse, tape, checkpoint, and working-set preflight.

## Evidence levels

The phase diagram is a structural capacity prediction, not proof that every
green TN point is production-ready.

Measured evidence currently available:

- A 12q, 64-slice explicit TN gradient workload achieved 7.29x speedup on
  eight A800s with numerical agreement.
- A 36q 4x9 complex128 checkpointed-reverse preflight passes with 128 slices,
  a 4 GiB intermediate limit, 16 GiB checkpoint budget, and 64 GiB rank-local
  working-set budget.  The predicted working set is 47.27 GiB versus an
  estimated 48.61 GiB full tape.

Why that is insufficient:

- the 12q workload fits one device;
- it does not demonstrate a capacity-forced gradient;
- the 36q forward production result does not prove reverse tape capacity.

Therefore the next hard gate is a 36q-or-larger workload where SV capacity and
MPS bond both fail and 1/2/4/8 A800 checkpointed-reverse scaling remains
positive without full-tape materialization.  The capacity preflight portion of
this gate now passes; execution and scaling remain to be measured.

## Claim boundary

The selector may recommend TN structurally, but production execution must fail
closed unless the sliced reverse joint preflight validates slice economics,
checkpoint memory, rank ownership, and the requested gradient contract.
