# Finite-field polynomial numerical encoding

Simulation owns `FiniteFieldPolynomialEncoding(degree, field)`, the numerical
counterpart of pinned Qualtran v0.7.0 `_GFPoly`. The field argument is the existing
`FiniteFieldEncoding`; polynomial and field objects remain inside Simulation's
optional Galois boundary. Compiler continues to own polynomial wire layouts and
physical decomposition. No new Core descriptor, executable admission, public
export or dependency is introduced.

```python
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

encoding = FiniteFieldPolynomialEncoding(1, FiniteFieldEncoding(2, 2))
polynomial = encoding.from_bits([0, 1, 1, 0])
assert encoding.to_bits(polynomial) == [0, 1, 1, 0]
```

The codec retains the unmodified source call chain: Cartesian field-element
products construct the polynomial domain; descending coefficients are left-padded
to degree plus one; bit encoding uses the field's array encoder; bit decoding
reshapes coefficient words then uses the existing field decoder. Validation
checks polynomial class, nominal coefficient-field identity and degree, in that
order. Array codecs reuse `value_encoding` default vectorization; array validation
applies scalar validation in flattened order. No second field algebra or binary
packing algorithm is implemented.

Construction and width remain lazy. Negative and symbolic degrees are not
normalized. Domain iteration, reshape, polynomial construction and array type
inference retain their source failures and diagnostics. Coefficient conversion
methods do not add a degree or length guard. Odd fields retain invalid binary
word failures rather than reducing them modulo the field order. The subsequent [numerical metadata connection](FINITE_FIELD_POLYNOMIAL_METADATA.md)
projects concrete declarations into generic domains, codecs and binding.
This numerical object alone does not authorize a polynomial Program port.

## Verification and resource accounting

Tests use the unchanged source at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Six small field/degree cases cover GF2,
GF4, GF8, GF3 and GF9, each through both original QGFPoly and CGFPoly numerical
interfaces. Complete domains, every small binary input word, coefficient padding,
scalar codecs, scalar/vector/matrix/empty array defaults, wrong fields/types,
over-degree values, malformed lengths, symbolic degrees and negative-degree
behavior are compared directly. Failures compare both category and exact text.

The existing GF2 polynomial addition path consumes the native coefficient
conversion and decoding. All 256 input pairs build its complete complex matrix.
The unchanged source decomposition is compared gate by gate with native lowering.
`circuit.append(op)` on a complex superposition agrees with the matrix and a
second append restores the input. The spectator remains zero.

Actual physical primitives are **4 CX**, zero T and zero measurements; no quantum
work bits are allocated. The independent source QEC estimate is **4 Clifford**,
zero T and zero measurements. These categories are recorded separately. This
coherent circuit has one deterministic branch, so there are no measurement
histories to enumerate. Numerical codecs and coefficient conversion emit zero
primitives. No odd-characteristic physical decomposition is claimed.

G2 remains open, including executable polynomial graph ports and source-wide register
integration. The source files are unchanged; G3 is not started.

Local checks: 79 new cases, 3886 concentrated regressions, 84 CPU/TN tests and two original upstream polynomial tests passed. Strict typing of two Python files, Black, Ruff, Simulation scope, architecture, internal IR and API checks passed. New codec coverage is 100% (50 statements, 10 branches). Nine locked source blobs matched the inventory.

Default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings in 134.21s (0:02:14)`. All observed failures belong to the seven-case baseline; 0 baseline cases were not observed before the configured stop. This is not a green full suite. Evidence uses prefix `g2-field-polynomial-encoding-` in the local reference directory.
