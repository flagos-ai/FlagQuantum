# Jacobi-Anger full call-chain audit

Simulation owns the numerical behavior adapted from Qualtran v0.7.0
`linalg/polynomial/jacobi_anger_approximations.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Both native owners preserve source
copyright. This batch fills default dependency calls, not a new approximation
algorithm. Unmodified source Git blobs and SciPy source/metadata hashes are
recorded independently in workspace evidence.

## Entire defined surface

| Original definition | Native responsibility and retained order |
| --- | --- |
| `approx_exp_cos_by_jacobi_anger` | `jacobi_anger_coefficients.py`; ascending order array, one vector jv call, original complex phase multiplication |
| `approx_exp_sin_by_jacobi_anger` | `jacobi_anger_coefficients.py`; same order array, one jv call, original actual dtype |
| `degree_jacobi_anger_approximation` | `jacobi_anger_degree.py`; symbolic C_JA expression or doubling/isclose/bisect/final assertion |
| Original `test_exp_cos_approximation`, `test_exp_sin_approximation` | 24 parameterized original tests run unchanged, and all 24 run again with only their helper bindings replaced by native defaults |

The implementation file has no other defined functions, classes, exported
constants or assets. The source test file has only those two parameterized
test functions. Symbolic expression classification and formulas reuse the
existing Core and SymPy boundaries. No Core contract, IR, public export or
protected API is changed.

All three native functions now accept omitted Bessel providers. They use lazy
`scipy.special.jv` imports for numerical work and retain the existing injection
path. Coefficients compute the original order array before Bessel evaluation;
cosine uses the literal `1j**orders * values` expression. There is no dtype cast,
input repair or new iteration cutoff. Scalar search preserves every queried
order, even its original final assertion calls. Symbolic degree evaluation and
module import require no SciPy import.

## Optional Bessel dependency review

SciPy **1.17.1** supplies the reference jv implementation. Simulation owns this
optional numerical dependency; it is imported only at the default numerical
call boundary, not installed by FlagQuantum and not added to the core manifest.
SciPy's metadata records its BSD redistribution license and bundled-component
notices. No SciPy code or binaries are vendored by this batch. Version, source
hashes and license metadata hash accompany the unmodified Qualtran blobs in
workspace evidence. This uses the same optional dependency decision as the
[FFT default call chain](QSP_INSPECTION_CLOSURE.md).

The existing scalar/vector provider signatures are the replacement interface;
finite cached Bessel tables replace jv without altering its consumers. An exit
would require a separately validated evaluator with the same dtype, rounding,
warning and failure behavior. No home-grown special function or fallback is
introduced. Missing SciPy produces ModuleNotFoundError, and a supplied provider
bypasses the import. Provider errors propagate. The symbolic path continues
without SciPy. Native keyword names and optional injection remain a deliberate
FlagQuantum interface adaptation, not an upstream signature compatibility claim.

## Bounded exceptional-input observations

Each original and native probe runs in its own child process with a 15-second
parent timeout. The only instrumentation records jv orders and delegates to
the unchanged SciPy function. No source search cutoff or return value is
injected. All 16 processes exited naturally before timeout in this environment.
Warnings, every Bessel order and exact outcomes matched between implementations.

| Input, with t=2 or precision=1e-5 unless specified | Observed source and native behavior |
| --- | --- |
| t=NaN, +infinity, -infinity | 1,025 jv visits; OverflowError: int too large to convert to float |
| precision=-1, NaN, -infinity | 19 jv visits; degree 162, due to exact Bessel underflow and original isclose behavior |
| precision=0 | 19 jv visits; degree 162 |
| precision=+infinity | Three jv visits; original AssertionError |

These are measured boundaries for the locked environment. No finite set proves
termination for every input, SciPy version or injected provider. The source
still has no nonfinite guard or iteration budget. Coarse-precision assertion
failure and the heuristic, non-certified truncation criterion remain inherited
limitations; they are not repaired or used as a global error guarantee. Under
the frozen rule, implementation parent **Q-0764 remains open**. Test parent
**Q-0765 closes** because its entire original surface passes with native bindings.

## Runnable default path

With SciPy installed by the caller:

```python
import numpy as np
from flagquantum.simulation.numerics.jacobi_anger_coefficients import approx_exp_cos_by_jacobi_anger
from flagquantum.simulation.numerics.jacobi_anger_degree import degree_jacobi_anger_approximation

degree = degree_jacobi_anger_approximation(2.0, precision=1e-7)
assert isinstance(degree, int)
coefficients = approx_exp_cos_by_jacobi_anger(2.0, degree=degree)
point = np.exp(0.3j)
value = np.polynomial.polynomial.polyval(point, coefficients) * point**(-degree)
np.testing.assert_allclose(value, np.exp(2j * np.cos(0.3)), atol=2e-7, rtol=0)
```

Normal insertion remains `circuit.append(op)`. Existing cached/source/native
replacement consumers append H/Rz, preserve the exact two instructions, compare
full complex polynomial matrices and evolved amplitudes, and verify their idle
clean wire. There is no measurement in these numerical helpers; enumeration of
measurement branches does not apply.

| Quantity | Emitted / separately modeled result |
| --- | --- |
| Added primitives / work / classical slots / measurements | 0 / 0 / 0 / 0 |
| Upstream QEC model | Not defined for these numerical functions |
| Classical coefficient data | 2*degree+1 entries for nonnegative degree |
| Classical search work | Recorded source Bessel visit sequence, independent of circuit counts |
| Symbolic result | Original positive unspecified C_JA asymptotic expression |

## Validation and evidence

63 added cases plus the existing 44 give 107 focused passing cases. Both owners
have 100% statement and branch coverage (42 statements, ten branches). The
concentrated numerical/tensor regression has 514 passing cases. All 24 original
source tests pass, and their full 24-case native binding replacement passes.
Strict typing covers both owners and all three test files; formatting, lint,
Simulation scope, architecture, IR and protected API checks pass. The default
suite is reported separately in the integration ledger, including known failures.

Evidence prefix: `ftqc-reference-env/g2-jacobi-anger-source-audit-*`, including
`boundaries.json` with all 16 actual process results, every jv order and warnings.
This is a single-device numerical inspection path; it claims no synthesis,
PyTorch differentiation, distributed execution or hardware performance. G2
remains open.

Default result: 6 failed, 21485 passed, 300 skipped, 1037 warnings in 138.74s (0:02:18). All six observed failures are known
baseline IDs. A targeted supplemental run reproduces the seventh baseline
failure, which maxfail did not reach. There are no new observed failures,
and no unobserved baseline ID after the supplement. This is not a full green
default-suite result.
