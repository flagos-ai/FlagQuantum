# QSP inspection source audit and remaining defect

Simulation owns the numerical behavior of Qualtran v0.7.0
`linalg/polynomial/qsp_testing.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its five defined functions are
mapped below. Copyright stays in the native owners; unchanged source blobs
are verified against the frozen inventory. Source tests and dependency hashes
are recorded separately. No upstream source is patched.

## Entire defined source surface

| Original function | Native owner and exact retained behavior |
| --- | --- |
| `random_qsp_polynomial` | `qsp_sampling.py`; one magnitude draw, optional phase draw, exactly degree coefficients, same RNG advancement and errors |
| `check_gqsp_polynomial_pair_on_random_points_on_unit_circle` | `qsp_sampling.py`; Polynomial constructors before random draw, original allclose and failure order |
| `_polynomial_max_abs_value_on_unit_circle` | `qsp_grid.py::polynomial_max_abs_value_on_unit_circle`; lazy SciPy import, asarray, dtype-preserving padding, one FFT and NumPy maximum |
| `scale_down_to_qsp_polynomial` | `qsp_grid.py`; outer asarray, list conversion, shared maximum, divide only above one and unchanged identity otherwise |
| `assert_is_qsp_polynomial` | `qsp_grid.py`; exact source finite-grid <=1 assertion and diagnostic |

There are no other functions, classes, exported constants or per-file assets.
Native keyword names express FlagQuantum semantics; the extra `fft` keyword is
optional, so ordinary default calls follow the original dependency chain. This
is semantic fusion, not an upstream-name alias package or signature snapshot.

The former Python float conversion has been removed. float16/32/64/longdouble,
complex64/128/clongdouble, bool, integer and object inputs are tested with the
same SciPy FFT; NumPy scalar types and precision, result arrays, warning order
and errors match the source. No input conversion beyond the source asarray is
added. ndarray subclasses and masked arrays follow source conversion/storage
behavior. Polynomial reconstruction retains the source's nested-object failure;
shape-only descriptors do not fabricate coefficients. Native descriptor errors
retain `SymbolicArrayShape` instead of the source `Shaped` class name, with the
same failure type and stage. That naming difference is explicitly not claimed
to be a byte-identical upstream diagnostic. Both Generator and RandomState
runtime sampling paths retain their original results, failures and next draws.

## Optional FFT dependency decision

The default call now lazily imports `scipy.fft.fft` at the same point as the
source. This fills a concrete missing numerical call chain without duplicating
an FFT implementation or changing the dependency manifest. SciPy is not added
to the core installation and is not downloaded by this module. A caller may
provide the existing FFT callback instead. If SciPy is absent, the original
ModuleNotFoundError is propagated; no fallback changes precision or algorithm.
Import order is tested before the shared helper's conversion, and after the
scaling wrapper's outer conversion.

Simulation owns this optional dependency boundary. SciPy is a mature CPU
numerical library under its BSD redistribution license; the reference metadata
contains the SciPy/Enthought copyright and redistribution terms, with bundled
component notices in its LICENSE file. This change imports the installed
library, does not vendor its code, and records version **1.17.1** and dependency
source hashes for the reference environment. No new native binary or manifest
entry is introduced. The replacement interface is the existing FFT callable;
its cached-transform replacement test already proves unchanged consumers. The
exit condition is a separately validated FFT implementation with matching dtype,
rounding and failure semantics; no such replacement is claimed here.

Importing the numerical modules does not import SciPy. Providing a callback also
bypasses SciPy import. Missing dependency, array-conversion and FFT callback
failures remain observable. Internal unprotected functions gain optional FFT
behavior; public exports, protected schemas and Core IR remain unchanged.

## Runnable default path

With SciPy installed in the caller's environment:

```python
import numpy as np
from flagquantum.simulation.numerics.qsp_grid import (
    scale_down_to_qsp_polynomial, assert_is_qsp_polynomial,
)

scaled = scale_down_to_qsp_polynomial([1.0, -2.0], n_points=8)
np.testing.assert_array_equal(scaled, [1 / 3, -2 / 3])
assert_is_qsp_polynomial(scaled, n_points=8)
```

Normal circuit insertion stays `circuit.append(op)`. Existing replacement
consumers preserve H/Rz instructions, full complex numerical evolution and the
idle clean wire. These are inspection helpers, not circuit synthesis. There is
no measurement in this path, so measurement-branch enumeration does not apply.

| Quantity | Evidence / separate source model |
| --- | --- |
| Numerical-helper emitted gates / work / slots / measurements | 0 / 0 / 0 / 0 |
| Upstream QEC cost | No model defined for these functions |
| Actual numerical calls | One or two uniform draws; pair evaluation; one FFT per maximum |
| Classical storage | O(degree + n_points), actual peak not benchmarked |

## Validation and remaining parent blocker

70 added cases plus the existing 44 give 114 focused passing cases, with both
owners at 100% statement and branch coverage (47 statements, six branches).
The concentrated numerical/tensor regression has 451 passing cases. All 16
unchanged original FFT-complementary tests pass (15 circuit tests deselected;
six pre-existing upstream marker warnings). Two original complementary consumers
and one Jacobi-Anger grid consumer also execute unchanged with native helper
bindings. This does not reimplement their algorithms or expand GQSP scenarios.
Strict typing covers both owners and all three test modules; format, lint,
Simulation scope, architecture, IR and protected API checks pass. The default
suite and source/dependency hashes are recorded in the workspace evidence
`ftqc-reference-env/g2-qsp-inspection-closure-*` and the integration ledger.

The inherited source TODO remains: four sample points can accept a polynomial
whose between-grid magnitude is 1.06. Native and source pass the same false
certificate. Random-point checks also certify only the sampled points. No
whole-circle guarantee, synthesis, hardware performance, gradient or distributed
execution is claimed. Under the frozen rule that unresolved source defects keep
parents open, **Q-0766 remains open**, despite completion of the interface and
call-chain audit. This does not close G2 or silently fix the upstream reference.

Default result: 7 failed, 21507 passed, 300 skipped, 1037 warnings in 162.55s (0:02:42). All seven observed failure IDs match the
previous baseline, with no new observed failure. The run stopped at maxfail;
it is not a full green default-suite result.
