# Matrix numerics: complete source-file closure

Simulation owns `numerics/random_hermitian.py` and `numerics/unitary_distance.py`.
Together they cover both functions in Qualtran v0.7.0 `linalg/matrix.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Production code does not import
Qualtran or Cirq. Copyright and pinned source links stay in each numerical owner.
This is a single-device CPU numerical inspection path, not a distributed kernel.

## Complete source surface

| Source surface | Native owner | Contract and acceptance |
| --- | --- | --- |
| `random_hermitian_matrix` | `random_hermitian.py` | Preserve Cirq RNG selection, one normal draw, QR, sign correction, symmetrization, failures and subsequent RNG draws |
| `unitary_distance_ignoring_global_phase` | `unitary_distance.py` | Preserve shape-check order, conjugate trace, clamp, warning behavior and NumPy scalar dtype/precision |
| `matrix_test.py::test_unitary_distance_zero` | `test_unitary_distance.py` | Both source dimensions and all three iterations run with the native metric substituted; unchanged original test also runs |

There are no classes, constants, exports, symbolic formulas, drawing hooks or
other functions defined in this source file. Source imports remain dependencies,
not extra FlagQuantum exports. Source blob identities are checked against the
frozen integration inventory. Cirq 1.7.0 is the observed reference dependency;
its `random_orthogonal` and `parse_random_state` source identities are recorded
separately. This does not claim reproducibility across BLAS/NumPy versions.

The sampler now includes the entire source RNG selection call chain. A Python
integer, including bool, initializes RandomState; None uses NumPy's global RNG;
a custom `randn` provider is retained. Source failure semantics are preserved
for invalid seeds, NumPy integers, Generator, missing providers, negative
dimensions, QR failure and degenerate QR. See [sampling details](RANDOM_HERMITIAN.md).
The default argument remains required, just as in Qualtran.

The distance metric retains NumPy scalar precision instead of coercing to a
Python float. Tests compare both scalar type and value for bool/integer,
float16/32/64/longdouble, complex64/128/clongdouble and object arrays. Empty and
nonfinite matrices retain source warnings and clamp behavior. Input ndarray
dtype is a NumPy boundary: annotations do not introduce input conversion or
unitarity checks absent from the source.

## User workflow and resources

```python
import numpy as np
from flagquantum.simulation.numerics.random_hermitian import random_hermitian_matrix
from flagquantum.simulation.numerics.unitary_distance import unitary_distance_ignoring_global_phase

hamiltonian = random_hermitian_matrix(4, 17)
np.testing.assert_array_equal(hamiltonian, hamiltonian.conj().T)
assert unitary_distance_ignoring_global_phase(np.eye(4), np.eye(4)) == 0
```

These helpers inspect dense numerical matrices. Existing circuit insertion stays
`circuit.append(op)`. The unchanged sampler consumer in
`test_random_hermitian.py` appends H/Rz, then compares phaseful numerical evolution
with both source and native sampler, including its clean spectator. There is
no measurement in this workflow, so measurement-branch enumeration does not apply.
Distance alone never certifies complex amplitudes or controlled equivalence.

| Quantity | Measured / source contract |
| --- | --- |
| Circuit instructions added by these helpers | 0 |
| Work qubits / classical slots / measurements | 0 / 0 / 0 |
| Upstream QEC model | Not defined for these numerical functions |
| Matrix storage | O(N**2); actual peak memory not benchmarked |
| Classical numerical work | Sampler QR O(N**3); distance dense product O(N**3) |

## Validation

66 focused cases pass, including 29 added closure cases. The concentrated
numerical/tensor consumer regression has 322 passing cases. Both unchanged
original matrix tests pass. Strict typing covers both owners and both test files;
format, lint, Simulation scope, architecture, IR and protected API checks pass.
Full default results are recorded separately in the integration ledger, including
existing failures; focused results do not imply a green default suite.

Evidence logs: `ftqc-reference-env/g2-matrix-closure-*` beside the workspace.
Parent closures are Q-0757 and Q-0758. This closes the matrix module only, not G2
or other numerical modules. Upstream sources were not modified.
