# Dense matrix comparison assertions

Simulation owns `numerics/matrix_assertions.py`, adapted from unchanged Qualtran
v0.7.0 `linalg/testing.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The phase-insensitive assertion
calls the existing native `unitary_distance_ignoring_global_phase`; no trace
metric is duplicated. Copyright is preserved. No new dependency, execution
engine, Compiler/Runtime import, protected API, or public export is introduced.

## Short local path

```python
import numpy as np
from flagquantum.simulation.numerics.matrix_assertions import (
    assert_matrices_almost_equal,
    assert_unitaries_equivalent_upto_global_phase,
)

identity = np.eye(2, dtype=np.complex128)
assert_matrices_almost_equal(identity, identity.copy())
phaseful = np.exp(0.31j) * identity
assert_unitaries_equivalent_upto_global_phase(identity, phaseful)
try:
    assert_matrices_almost_equal(identity, phaseful)
except AssertionError:
    pass
else:
    raise AssertionError("phase-sensitive comparison must reject this phase")
```

The normal workflow remains `circuit.append(op)`. The source/native replacement
test appends an existing H/Rz operation, checks its instruction stream, compares
the complete complex state against the independent phaseful formula, and checks
its idle clean wire. The consumer accepts either matrix assertion without code
changes. No global-phase alignment is used in that phase-sensitive comparison.

## Retained source behavior

| Assertion | Criterion | Source failure order |
| --- | --- | --- |
| `assert_matrices_almost_equal` | `np.linalg.norm(left-right) <= atol` | Equal-shape assertion before subtraction; then default-norm assertion |
| `assert_unitaries_equivalent_upto_global_phase` | Shared trace-overlap distance <= atol | Shared same-shape/square ValueErrors; then assertion with source diagnostic |

For two-dimensional arrays, the first criterion is the Frobenius norm, not a
spectral or entrywise bound. It accepts rectangular matrices and vectors as the
source does; it does not require square matrices or unitarity. The typed native
inputs cover float32/float64/complex64/complex128 NumPy arrays. Input arrays are
not modified. A norm exactly equal to atol passes. Empty matrices, negative
atol, NaN and shape failures are checked against the original.

The unitary criterion assumes caller-provided unitarity and retains the
source metric's nonnegative clamp. For example it accepts identity versus
2*identity even though the second is not unitary. This is an explicitly retained
source assumption, not validation of nonunitary equivalence. Global phase is
ignored only in this named assertion. Once the phaseful operation is controlled,
it can become a relative phase and fail even this assertion. Tests retain that
counterexample and also show that an entrywise/spectral tolerance can pass while
the source Frobenius criterion fails.

These are assertions for inspection/testing: they retain Python assertion
semantics, including removal under Python optimization. They are not runtime
capability validation, finite-input normalization, or a physical acceptance gate.
Foreign/symbolic arrays and other dtype behavior are outside this typed slice.
No reference source is changed to obtain parity.

## Verification and costs

Direct source comparisons cover boundary tolerances, multiple shapes, complex
phase, relative phase, original assertion messages and shared shape errors.
The original generalized-QSP consumer is also run at selected small degrees
without modifying its helper code; this tests original assertion use and does
not claim its whole synthesis algorithm was newly ported in this batch.

| Quantity | These assertions |
| --- | --- |
| Added physical gates / work wires / measurements | 0 / 0 / 0 |
| Source QEC physical resource model | None |
| Matrix comparison | One dense difference and default norm |
| Unitary comparison | Existing trace-overlap matrix product |
| Numerical storage | O(N**2) for square matrices; peak not measured |

The replacement circuit's original two H/Rz gates are unchanged. Neither a norm
nor an assertion is a synthesized primitive, gate-count estimate, measurement
branch, gradient implementation, or scalability claim. Finite typed comparison
acceptance does not close full-parent or full G2 acceptance.
