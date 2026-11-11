# Polynomial and matrix assertion source closure

Simulation owns the NumPy inspection functions in `numerics/unitary_polynomial.py`
and `numerics/matrix_assertions.py`. This batch closes Qualtran v0.7.0
`linalg/polynomial/basic.py`, its `basic_test.py`, and `linalg/testing.py`, pinned
to `bcb17bb25360ed68ce3bc084645db3e929799389`. Source copyright remains intact.
The original modules are not edited. Production code adds no dependency on
Qualtran or Cirq and no public exports or protected API changes.

## Entire source surface and interfaces

| Source file / defined surface | Native target and preserved semantics |
| --- | --- |
| `basic.py::evaluate_polynomial_of_unitary_matrix` | `unitary_polynomial.py`; ascending coefficients, square assertion, integer powers, negative offset uses adjoint powers, dtype-preserving source `+=`, one subsequent matrix multiply per coefficient |
| `basic_test.py::test_evaluate_polynomial_of_unitary_matrix` | `test_unitary_polynomial.py`; original test function executes unchanged with only its evaluator binding replaced; original source test also executes separately |
| `testing.py::assert_matrices_almost_equal` | `matrix_assertions.py`; equal-shape assertion followed by default norm assertion, without normalization or extra validation |
| `testing.py::assert_unitaries_equivalent_upto_global_phase` | `matrix_assertions.py`; shared native trace distance, exact original assertion diagnostic and tolerance |

There are no other defined functions, classes, constants, exports or source
assets in these three files. The testing helper has no separate original
`testing_test.py` inventory entry. Its two functions are instead compared
directly to the unchanged source. The phase-insensitive assertion depends on
the previously closed [matrix source functions](MATRIX_NUMERICS.md).

The array boundary now follows NumPy rather than limiting annotations to four
dtypes. Integer, unsigned, bool, float16/32/64/longdouble, complex64/128/
clongdouble and object cases are checked. A dtype that raises in the original
also raises with the same type and message in the native implementation; this
includes bool subtraction and unsigned negative-coefficient OverflowError.
Object arrays may contain symbolic values without a defined numerical norm.
Their source failures are preserved, not converted into numeric support.

NumPy ndarray subclasses, masked arrays and matrix subclasses exercise the
polynomial path. Assertions check ndarray subclasses, masked arrays and symbolic
object arrays. Results, return types, warnings and failures are compared; no
coercion or subclass repair is inserted. The source's Python assertions remain
assertions, including removal under Python optimization. Shape and power errors
occur in source order. No unitarity or finite-input guard is added.

The polynomial accumulator uses the original `result += power * coefficient`
expression. The prior explicit `np.add(..., out=result)` was equivalent on the
previously tested ordinary arrays; this batch restores the literal operation
and verifies the broader NumPy dispatch boundary without claiming a newly found
value discrepancy on those old cases.

## Runnable workflow and measured costs

```python
import numpy as np
from flagquantum.simulation.numerics.unitary_polynomial import evaluate_polynomial_of_unitary_matrix
from flagquantum.simulation.numerics.matrix_assertions import assert_matrices_almost_equal

unitary = np.array([[0, -1j], [1j, 0]])
actual = evaluate_polynomial_of_unitary_matrix([1, 0, 1], unitary, offset=-1)
assert_matrices_almost_equal(actual, 2 * unitary)
```

Normal operation insertion remains `circuit.append(op)`. Existing source/native
replacement consumers append H/Rz, compare the original instruction stream and
full complex matrix/state, and check their idle clean wire. There is no
measurement in these numerical helpers; exhaustive measurement branches do not
apply. A phase-insensitive assertion cannot replace phaseful amplitude checks.

| Quantity | Emitted or modeled result |
| --- | --- |
| Physical primitives added by the numerical helpers | 0 |
| Work wires / classical slots / measurements | 0 / 0 / 0 |
| Upstream QEC model for these functions | Not defined |
| Polynomial numerical work | Initial matrix power plus one dense product per coefficient |
| Matrix assertion work | Difference and default norm, or shared trace-overlap product |
| Dense numerical storage | O(N**2); actual peak memory and execution timing not benchmarked |

This is a single-device CPU numerical inspection path. No gradient, synthesis,
distributed execution or hardware-performance claim is made.

## Acceptance evidence

The two focused suites have 113 passing cases, including 59 added closure cases.
The concentrated numerical/tensor regression has 381 passing cases. Both
owners have 100% statement coverage (26 statements); the polynomial owner
also covers all four branches. The assertion module has no coverage.py
branches. The original basic test runs both unchanged and with the native
evaluator substituted. Evidence and concentrated numerical/tensor regression are recorded
under `ftqc-reference-env/g2-polynomial-assertion-closure-*` in the workspace.
Strict typing covers both owners and both test files. Default suite failures
remain separately reported in the integration ledger; focused success is not
a claim of whole-repository acceptance.

Parent closures: Q-0762, Q-0763 and Q-0767. This batch does not close Jacobi-Anger,
QSP sampling, their dependency adaptations, or G2 as a whole.

Implementation commits: polynomial `59fc7cdc`, assertions `8da278a9`.
Default: 7 failed, 21507 passed, 300 skipped, 1037 warnings in 189.68s (0:03:09). All seven observed failure IDs match the prior
baseline, with no new observed failure; maxfail stopped the run, so this is
not a full green default-suite result.
