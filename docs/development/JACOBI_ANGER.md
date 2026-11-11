# Jacobi-Anger Laurent approximation

Simulation owns two independent numerical responsibilities:
`numerics/jacobi_anger_coefficients.py` generates ascending coefficients for
orders -degree through +degree, and `numerics/jacobi_anger_degree.py` reproduces
the source's threshold search and symbolic asymptotic expression. Both adapt
Qualtran v0.7.0 `linalg/polynomial/jacobi_anger_approximations.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`, preserving its copyright notice.

## Optional dependency boundary

Both numerical owners now accept an optional Bessel provider. When omitted,
they lazily import `scipy.special.jv`, the same evaluator called by the original
source. This is a caller-installed optional dependency, not an automatic install
or a core dependency manifest change. Missing SciPy and provider errors
propagate; there is no substitute implementation or precision fallback.

An explicit vector provider takes `(orders: ndarray[int64], t: float)`; a scalar
provider takes `(order: int, t: float)`. Tests retain cached-provider replacement
without changing coefficient, degree, matrix or circuit consumers. The symbolic
degree path bypasses Bessel import and reuses existing SymPy support and Core's
`is_symbolic_value` classifier. Default calls now work without the native-only
required-provider adaptation. Native keyword names and the optional injection
argument remain FlagQuantum-owned; this is not an upstream alias package.

See [the complete source audit and optional dependency review](JACOBI_ANGER_SOURCE_AUDIT.md).

## Executable numerical composition

The example requires a caller environment containing SciPy. SciPy is supplied by
the comparison environment; the production default now imports it lazily when
needed. The example keeps an explicit provider to demonstrate replacement.

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
  implementation retains the literal original `1j**orders * values` expression.
- Sine coefficients are `J_n(t)`: their actual source dtype is float64 even
  though the source annotation says complex128. Native typing follows the data.
  Both use one full-array Bessel call; degree -1 retains the source empty array.
  Complex t follows the original SciPy output dtype rather than the real-input
  return annotation; no runtime cast is added.
- Degree search doubles from 1, uses NumPy `isclose` with the original defaults,
  then `bisect_left` and the original final assertion. Tests compare every
  scalar Bessel argument in visitation order, including final assertion calls.
  Zero t returns degree zero; t=1 with precision=1 retains the source assertion
  failure. No correction is made to the reference.
- Any SymPy parameter selects the original expression with positive unspecified
  `C_{JA}`. The expression is equal to the unchanged source, including a SymPy
  integer/rational input. It is not a concrete certified degree or gate count.
- Concrete search with no evaluator now calls the same SciPy jv by default.
  The prior native-only ValueError has been removed. With the locked SciPy,
  NaN/infinite t produces 1,025 doubling visits and then OverflowError; some
  invalid precisions underflow to an exact zero Bessel value and return 162.
  All eight audited boundary cases exit naturally within a 15-second process
  budget, with identical source/native call sequence, warnings and outcomes.
  These finite observations are not a termination guarantee for arbitrary
  inputs, providers or dependency versions. No production guard is added to
  the unchanged source search. Its coarse-precision assertion and heuristic
  limitations remain open and are not an error certificate.


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
