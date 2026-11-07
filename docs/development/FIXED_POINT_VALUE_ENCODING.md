# Fixed-point stored integer encodings

This numerical slice belongs to Simulation. It follows the unchanged `_Fxp`
methods in Qualtran v0.7.0 `data_types.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interface and ownership

Functions take the stored integer value or big-endian bit array, total width,
fractional width and signedness. They return values or bits, allocate no quantum
work wires and emit no instructions. There is no new dtype hierarchy or Core
contract. Compiler remains the owner of gate decomposition and Runtime of
execution; the codecs import neither.

The source raw representation is the same-width signed or unsigned integer.
Fractional width affects shape constraints, integral-width metadata and conversion
of a stored word to a real scale, but does not change the bit pattern of that
stored integer. This is not a float-to-word quantizer.

The implementation must reuse existing integer scalar validation/codecs and
shared default array vectorization. In particular, `_Fxp` arrays use the default
scalar codec path even for narrow unsigned words; the `_UInt` optimized mask path
would change invalid-value behavior. Array validation similarly uses flattened
scalar checks. Source numeric types, zero/negative or symbolic shape diagnostics
and evaluation order must remain observable.

The external `fxpmath.Fxp` template, real quantization and real-value domain are
separate, unaccepted paths. They must not be replaced by guessed float rounding
or described as delivered by this stored-integer slice. No production dependency
is added here.

## Delivered numerical paths

`fixed_point_encoding.py` exposes shape validation, integral-width subtraction,
the raw integer domain, scalar/array validation and codecs, and the unchanged
`value / 2**fractional_width` real-scale helper. The latter does not check that its
input lies inside the raw integer range and does not quantize real inputs.
Shape validation happens before returning a domain iterator; subsequent integer
domain arithmetic remains lazy, as with source construction followed by iteration.

```python
import numpy as np
from flagquantum.simulation.fixed_point_encoding import (
    fixed_point_array_from_bits,
    fixed_point_array_to_bits,
    fixed_point_real_value,
)

words = np.array([-4, -1, 0, 3])
bits = fixed_point_array_to_bits(words, 3, 2, signed=True)
np.testing.assert_array_equal(
    fixed_point_array_from_bits(bits, 3, 2, signed=True), words
)
assert fixed_point_real_value(-1, 3, 2, signed=True) == -0.25
```

The golden path runs with:

```sh
python -m examples.algorithms.classical_fixed_point_values
pytest -q tests/unit/test_fixed_point_encoding.py tests/unit/test_register_phase_gradient.py
```

Tests compare the original `_Fxp` and public quantum/classical types across 18
shape configurations, including symbolic and NumPy dimensions. Raw validation,
bit order, result types, array inference, malformed words and source errors remain
unchanged. A replacement test substitutes the existing integer codec functions
without changing the fixed-point consumer or reference.

The executable example uses `circuit.append(op)` for the existing three-bit
phase-gradient decomposition with a spectator. The tests compare every complex
amplitude with the unchanged source tensor and inverse restoration. The emitted
IR has three `phase` instructions, no explicit T instructions, measurements or
work wires. The source logical model has two Clifford gates and one T. These
different counting bases are reported separately; scalar value encoding adds no
quantum gates. Existing register phase-gradient tests cover its source instruction
sequence and measured controlled variants, rather than adding a second emitter.

Acceptance: 422 new cases and 1,462 concentrated passes without skips. The 15
existing numerical warnings remain visible. Strict typing of the new module and
example, formatting, ownership and architecture checks pass. Four pinned source
blobs for data types, phase gradients, rotation gates and gate budgets are checked
against the immutable inventory; none is modified.
