# Random real symmetric inspection matrices

Simulation owns `numerics/random_hermitian.py`. It adapts Qualtran v0.7.0
`linalg/matrix.py::random_hermitian_matrix` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389` and inlines the small numerical body
of its Cirq `random_orthogonal` dependency. Both copyright notices are preserved.
No Compiler, Runtime, Qualtran, or Cirq production dependency is added.

The dependency observed in the reference environment is Cirq **1.7.0**, not a
transitive version pinned by Qualtran. Its `testing/lin_alg_utils.py` SHA256 is
`dac407b6d364998f2122cf2e5dad3d24e68b1a080b6985f6bb7ce18c5c354769`.
The comparison records that version and file hash separately from Qualtran's
pinned Git blob. Exact numerical equality is evidence for this environment;
it is not a cross-BLAS or cross-NumPy bitwise reproducibility guarantee.

## Local example

```python
import numpy as np
from flagquantum.simulation.numerics.random_hermitian import random_hermitian_matrix

rng = np.random.RandomState(17)
hamiltonian = random_hermitian_matrix(4, rng)
np.testing.assert_array_equal(hamiltonian, hamiltonian.conj().T)
assert hamiltonian.dtype == np.float64
assert np.linalg.norm(hamiltonian, ord=2) <= 1 + 2e-15
values, vectors = np.linalg.eigh(hamiltonian)
evolution = (vectors * np.exp(-0.31j * values)) @ vectors.conj().T
np.testing.assert_allclose(evolution.conj().T @ evolution, np.eye(4), atol=1e-14)
```

This is a numerical inspection matrix and optional dense evolution, not circuit
synthesis. Ordinary operation insertion remains `circuit.append(op)`. The test
consumer appends an existing H/Rz operation, applies the numerical evolution,
and checks full complex amplitudes and its idle clean wire with either sampler.
The sampler replacement requires no consumer changes. No polynomial or
phase-insensitive distance is used to weaken amplitude comparison.

## Retained algorithm and boundaries

1. Consume exactly one `randn(dimension, dimension)` call on the supplied RNG.
2. Compute NumPy QR and the diagonal of R.
3. Multiply Q's columns by `diagonal / abs(diagonal)`.
4. Return `(Q + Q.conj().T) / 2`.

The result is real symmetric, hence Hermitian, with operator norm at most one
up to roundoff for ordinary finite draws. It is not a general complex Hermitian
ensemble, a unitary, or a density matrix. An explicit sampler remains caller-owned. The source Cirq parser is now
inlined: `None` consumes NumPy's global RNG, a Python `int` (including `bool`)
constructs a seeded `RandomState`, and other providers pass through unchanged.
NumPy integer scalars and Generator objects retain the source AttributeError
when they lack `randn`; invalid Python seeds retain NumPy's ValueError. There
is no validation or coercion that changes failure order.

Dimension zero returns a (0, 0) matrix. Negative dimensions retain NumPy's
ValueError before consuming random numbers. QR failure leaves the one normal
draw consumed. Degenerate zero diagonal retains division warnings and NaN;
there is no retry, clipping, silent repair, or distribution substitution.

## Evidence and resources

Twenty new tests compare dimensions 0, 1, 2, 3, 7 across three seeds and repeated
calls against unchanged Qualtran, including following RNG draws. Independent
checks cover symmetry, norm, dtype, global RNG, failure order, degenerate QR,
and a source/native replacement through the same phaseful consumer.
The original matrix tests do not test this sampler; their success alone is
therefore not claimed as sampler acceptance. Direct source comparisons supply
that evidence.

| Quantity | Numerical sampler |
| --- | --- |
| Added physical gates / work wires / measurements | 0 / 0 / 0 |
| Original QEC resource model | None for this numerical function |
| Numeric storage | O(dimension**2), multiple arrays; peak not measured |
| Numeric work | One normal draw, QR O(dimension**3), column correction, symmetrization |

These numerical operations are not physical primitive gate counts, and no
performance, scalability, PyTorch-gradient, or hardware-execution claim is made.
The original explicit-RNG slice was partial. The completed RNG selector and
whole-file closure evidence are in [Matrix numerics](MATRIX_NUMERICS.md). G2
as a whole remains open.
