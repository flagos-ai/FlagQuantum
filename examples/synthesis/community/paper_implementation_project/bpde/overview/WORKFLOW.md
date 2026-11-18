# Hydrogen BPDE execution and checks

The native molecular driver regenerates all original hydrogen geometries. It
projects onto the exact invariant paired-determinant sector and validates the
sector energy against the full-space FCI result. The Hartree–Fock state is the
second determinant. All three real Pauli coefficients, including the identity
coefficient, contribute to the relative branch phase.

Each scan applies the original second-order Strang formula with
`ceil(time / 0.5)` steps. The two bounded two-by-two products are raised to the
exact integer powers and packed into a single two-qubit controlled matrix.
This reduces simulator instructions while retaining every step of the source
approximation. The independent recurrence applies every small exponential
to the branch vector at every step. It checks every complex amplitude at each
executed offset, including the relative and global phase.

The source computes readout Z. The plots and Gaussian fit use
`2 * count_zero / shots - 1`, with the probability of zero recorded separately.
Each binding draws the source's one million Bernoulli trials at an explicit
local seed. A local SciPy Gaussian fit and the same precision-weighted update
preserve the five-offset grid, threshold 0.001, time rule, and 30-iteration
cap. The fit evaluation limit is explicit. The source delegates numerical
environment versions to its notebook runtime; the local seed and actual
observations are recorded with the results.
