# Bounded integer values

This slice belongs to Simulation and follows the unchanged `_BUInt`, `BQUInt`
and `BCUInt` numerical methods in Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interfaces and ownership

`bounded_integer_values.py` owns the default bound expression, shape validation,
domain and scalar/array value checks. The scalar/array checks depend only on the
exclusive bound; callers constructing a word first validate `(width, bound)`.
`bounded_integer_encoding.py` owns scalar/array big-endian bit representation.
Its entry points validate shape, then use the bound checks and existing unsigned
integer scalar codec. Array encoding uses the shared default scalar vectorization
boundary rather than the specialized `_UInt` array mask path.

No new dtype class, register descriptor or Core schema is introduced. Values and
arrays are numerical outputs. No quantum work wires or measurement slots are
allocated by these helpers. Compiler keeps ownership of bounded SELECT traversal
and its strict concrete wire/extent validation; Runtime owns execution. Neither
numerical module imports those layers.

Source distinctions to preserve include constructor symbolic short circuits,
the Python-int-only domain guard, negative-before-upper-bound scalar checks,
range-only array validation, decode followed by bound validation, and raw default
bound exponentiation without numeric casts. The source permits some diagnostic
shapes that cannot describe executable bounded address words; the compiler's
requirements must not be weakened to accept them.

The user example constructs coherent valid addresses through these codecs
and use existing bounded Pauli SELECT with `circuit.append(op)`. Source traversal,
every small measurement branch, complex phases and clean work are validated
separately from the numerical helper behavior and analytical resource model.

## Running the slice

```python
import numpy as np
from flagquantum.simulation.bounded_integer_encoding import (
    bounded_integer_array_from_bits,
    bounded_integer_array_to_bits,
)

addresses = np.array([0, 1, 2])
bits = bounded_integer_array_to_bits(addresses, 2, 3)
np.testing.assert_array_equal(bounded_integer_array_from_bits(bits, 2, 3), addresses)
```

```sh
python -m examples.algorithms.classical_bounded_integer_values
pytest -q tests/unit/test_bounded_integer_values.py tests/unit/test_bounded_integer_encoding.py tests/unit/test_bounded_pauli_select.py
```

The example executes all three valid addresses and both target values in coherent
superposition. Tests also include a spectator and force every measurement history,
comparing the complete complex state against both unchanged source instructions
and the independent Pauli action. Work returns to zero. The existing traversal is
not changed; source dependencies, operands and feedback are compared directly.

The two-bit, three-row example emits four T/T-dagger instructions, one measurement
and uses one work wire. Its analytical model separately reports four T equivalents,
one measurement and four peak logical qubits. Numeric codecs themselves emit no
gates. Padded-address behavior of the bounded quantum traversal remains covered
by existing tests and is not reinterpreted as a valid bounded classical value.

The numerical comparison includes 20 shape/bound configurations. Python-int-only
domains, NumPy and Sympy bound objects, default exponentiation, scalar versus
array checks, malformed decode lengths, empty vectorization and post-decode bounds
remain observable. A Sympy integer bound of nine bypasses a three-bit constructor
capacity guard, so a four-bit word can decode to eight; encoding that same value
still fails the underlying unsigned width check. This diagnostic source difference
is preserved without broadening the compiler's accepted address shapes.

Acceptance: 199 value tests and 325 encoding/consumer tests are new. Concentrated
regression passes 1,103 cases without skips; three existing numerical warnings
remain visible. Changed production/example files pass strict typing, formatting,
ownership and architecture checks. Pinned source files remain unchanged.
