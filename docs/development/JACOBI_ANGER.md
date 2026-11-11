# Jacobi-Anger Laurent approximation

Simulation owns two independent numerical responsibilities:
`numerics/jacobi_anger_coefficients.py` generates ascending coefficients for
orders -degree through +degree, and `numerics/jacobi_anger_degree.py` reproduces
the source's threshold search and symbolic asymptotic expression. Both adapt
Qualtran v0.7.0 `linalg/polynomial/jacobi_anger_approximations.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`, preserving its copyright notice.

## Explicit dependency boundary

FlagQuantum does not declare SciPy as a production dependency. The native
functions therefore require caller-owned Bessel evaluation for numerical work;
they do not silently import a transitive library or add a new package dependency.
The existing NumPy numerical inspection environment is still required. A vector
provider takes `(orders: ndarray[int64], t: float)` and returns real float64
Bessel values; a scalar provider takes `(order: int, t: float)` and returns a real
scalar. The reference tests and example supply `scipy.special.jv`, exactly as
used by the unchanged source. Tests also replace it with finite cached values
without changing coefficient, degree, matrix, or circuit consumers.

This is a deliberate native signature adaptation. It does not claim drop-in
upstream API compatibility or a built-in Bessel implementation. Provider errors
propagate. Providers own their dtype, shape, numerical accuracy and evaluation
cost; these are typed inputs rather than a new registry or implicit fallback.
The symbolic degree path needs no Bessel provider and lazily uses existing
SymPy support and Core's `is_symbolic_value` classifier.

## Executable numerical composition

The example requires a caller environment containing SciPy. SciPy is supplied by
the comparison environment, not installed or imported by these production modules.

```python
import numpy as np
from scipy.special import jv
from flagquantum.simulation.numerics.jacobi_anger_coefficients import (
    approx_exp_cos_by_jacobi_anger,
)
from flagquantum.simulation.numerics.jacobi_anger_degree import (
    degree_jacobi_anger_approximation,
)
from flagquantum.simulation.numerics.unitary_polynomial import (
    evaluate_polynomial_of_unitary_matrix,
)

degree = degree_jacobi_anger_approximation(2.0, precision=1e-10, bessel_jn=jv)
assert isinstance(degree, int)
coefficients = approx_exp_cos_by_jacobi_anger(2.0, degree=degree, bessel_jn=jv)
angles = np.array([-0.71, 0.23])
unitary = np.diag(np.exp(1j * angles))
result = evaluate_polynomial_of_unitary_matrix(
    tuple(complex(c) for c in coefficients), unitary, offset=-degree,
)
expected = np.diag(np.exp(2j * np.cos(angles)))
np.testing.assert_allclose(result, expected, atol=2e-10, rtol=0)
```

Normal operation insertion remains `circuit.append(op)`. Approximation data is
not an appendable synthesized circuit. Replacement tests append the existing
H/Rz operation and apply the resulting dense polynomial, checking full complex
amplitudes and its idle clean wire. There is no measurement in this numerical
slice; all-branch physical erasure claims are outside its scope.

## Retained source semantics and exceptions

- Cosine coefficients are `i**n * J_n(t)` with complex128 output. The typed native
  implementation spells the same multiplication ufunc as `np.multiply`.
- Sine coefficients are `J_n(t)`: their actual source dtype is float64 even
  though the source annotation says complex128. Native typing follows the data.
  Both use one full-array Bessel call; degree -1 retains the source empty array.
- Degree search doubles from 1, uses NumPy `isclose` with the original defaults,
  then `bisect_left` and the original final assertion. Tests compare every
  scalar Bessel argument in visitation order, including final assertion calls.
  Zero t returns degree zero; t=1 with precision=1 retains the source assertion
  failure. No correction is made to the reference.
- Any SymPy parameter selects the original expression with positive unspecified
  `C_{JA}`. The expression is equal to the unchanged source, including a SymPy
  integer/rational input. It is not a concrete certified degree or gate count.
- A concrete search with no evaluator raises a native ValueError; this extra
  boundary is separate from source parity. Finite supported inputs are tested.
  Nonfinite search inputs and arbitrary invalid precision have no source
  iteration guard; they remain unaccepted rather than being run without a bound.
  The doubling/bisect predicate is the source heuristic, not a new proof of a
  globally minimal truncation degree or a uniform tail-error bound.

## Verification and resource interpretation

Direct source comparisons cover coefficients, actual dtypes, empty arrays,
zero/negative t, degree visitation, formula assumptions, provider replacement,
and the original coarse-precision failure. Complete complex matrices are
compared to the original Laurent evaluator and independently to the exponential
on a diagonal unitary. Original source tests run unchanged and separately.

| Quantity | This numerical slice |
| --- | --- |
| Added physical primitive gates / work wires / measurements | 0 / 0 / 0 |
| Source QEC physical resource model | None in these functions |
| Numerical coefficient storage | 2*degree+1 entries for nonnegative degree |
| Numeric search cost | Source scalar provider call sequence, not a circuit cost |
| Symbolic result | Asymptotic degree expression with unspecified positive constant |
| Dense matrix consumer | Existing O(N**2) storage; no measured peak or performance claim |

The existing two H/Rz gates in the append consumer are unchanged. Numerical
Bessel calls, polynomial degree, and matrix products must not be reported as
measured quantum primitive counts. No synthesis, torch-gradient, scalability,
new execution engine, protected API, or full-G2 acceptance is claimed.
