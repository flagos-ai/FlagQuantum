# Scalar value encodings

Simulation owns numerical value encoding. Compiler owns reversible gate emission;
Runtime owns execution. These helpers do not introduce a register type hierarchy,
change Core schemas, or depend on Compiler.

`simulation/value_encoding.py` owns the callable boundary shared by scalar
codecs: `encode_value_array(values, encode)` applies the original scalar-to-vector
signature and uint8 conversion; `decode_value_array(bits, decode)` applies the
original vector-to-scalar signature and NumPy output inference. These are the
default `BitEncoding` array paths in unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

`simulation/integer_encoding.py` uses that boundary for signed and wide unsigned
default array paths. Its existing narrow unsigned mask path remains separate,
because the source does not apply scalar validation in that optimized path.
Empty arrays, scalar inputs, core-dimension errors and result dtype inference are
preserved rather than normalized with a new array policy.

```python
import numpy as np
from flagquantum.simulation.integer_encoding import (
    vectorized_integer_array_from_bits,
    vectorized_integer_array_to_bits,
)

values = np.array([-3, 0, 2])
bits = vectorized_integer_array_to_bits(values, 3, signed=True)
np.testing.assert_array_equal(
    vectorized_integer_array_from_bits(bits, 3, signed=True), values
)
```

The shared helper tests compare the unchanged upstream default array methods with
native vectorization of upstream scalar callbacks. Separate codec suites compare
native scalar semantics and existing integer consumers. Replacing the native
array boundary must change its consumers without changing them or patching the
reference implementation.

```sh
pytest -q tests/unit/test_value_encoding.py tests/unit/test_integer_encoding.py tests/unit/test_vectorized_integer_encoding.py
```

Run source comparisons in the pinned reference environment. Copyright and source
links remain in the corresponding production modules. This slice does not claim
that the complete upstream data type hierarchy has been ported.

## Single-bit values

`simulation/bit_encoding.py` ports `_Bit`, shared by the source quantum and
classical bit types. The validator tests equality to zero or one; it does not
require an integer type. Encoding validates and then applies `int`. Decoding
requires exactly one element and returns that element unchanged, including its
identity and type. Decoding is not a validator of binary values.

Array validation uses elementwise equality and `all`, including empty arrays.
Encoding and decoding use the shared default boundary, retaining NumPy shape and
dtype inference. The source scalar, array and malformed-value behavior is tested
against both `QBit` and `CBit` without changing either reference.

```sh
pytest -q tests/unit/test_bit_encoding.py
```

The coherent check compares all complex amplitudes of CNOT on two bit values
with the unchanged source tensor, including a spectator. Applying CNOT twice
restores the state. Its model is one Clifford, zero T and no work or measurements;
the value codecs themselves emit no quantum gates.

## Ones' complement values

`simulation/ones_complement_encoding.py` owns the `_IntOnesComp` numerical
methods shared by the upstream quantum and classical types. Words are big endian:
index zero is the sign; remaining bits hold the magnitude XOR the sign. Unsigned
magnitude encoding and decoding reuse `integer_encoding.py`. The array methods
reuse the callable boundary above. Inputs and scalar values are not silently
normalized to Python integers before source arithmetic.

The source constructor-width guard is shared with native register metadata in
`core/ir/program/register_types.py`. Numerical codecs and signed truncation use
that same function at their existing validation sites.

The port preserves these upstream distinctions:

| Method | Source behavior retained |
| --- | --- |
| Constructor guard | Rejects a Python integer width of one; no new nonpositive-width guard |
| Domain | Excludes both magnitude endpoints, from `-2**(width-1)+1` through `2**(width-1)-1` |
| Scalar validation | Includes both endpoints; magnitude encoding subsequently rejects them |
| Array validation | Flattens and checks scalars, rather than applying a range-only array rule |
| Decode | XORs the suffix before applying the sign power; does not repair length or validate bits |
| All-one word | Decodes the negative-zero representation to numeric zero |

Tests compare `QIntOnesComp` and `CIntOnesComp` at concrete, NumPy and symbolic
widths, including malformed inputs, output types, empty arrays and overflow
diagnostics. These diagnostic interfaces are not a claim of executable symbolic
quantum arithmetic.

The existing reversible ones-to-twos conversion now consumes the native codecs
in its tests while retaining explicit equality with the upstream value encoders.
Its physical implementation is unchanged. Source `SignedIntegerToTwosComplement`
declares an abstract Toffoli cost but has no root gate decomposition; only its
available component gate sequences are compared directly with upstream.

Scalar negative-zero decoding must not replace the reversible full-word mapping.
For width three, canonical zero and all-one negative zero cannot both map to the
same output in a reversible circuit. The established full-word extension maps
the latter to the otherwise unused two's-complement minimum. Existing tests cover
all small raw words and forced measurement histories, complete complex amplitudes
and work cleanup in both directions. Every wire outside the word is a work wire
owned by the existing conversion or an explicitly tested spectator.

```sh
python -m examples.algorithms.classical_signed_values
pytest -q tests/unit/test_ones_complement_encoding.py tests/unit/test_signed_encoding.py
```

The example uses `circuit.append(op)` and restores all complex amplitudes after
forward and adjoint conversion, including a spectator and zeroed work. It prints
the emitted four T instructions, one measurement and one work wire separately
from the source model's one Toffoli, zero explicit measurements and four T
equivalents. This is a model/decomposition distinction, not a claim that the
unavailable source root decomposition has been matched.

Batch acceptance: 490 new cases, 1,441 concentrated passes with no skips; 207
reference/native numerical diagnostic warnings retained. The pinned
`data_types.py` Git blob is unchanged. Production modules and the example pass
strict typing for the changed surface, formatting, ownership and architecture
checks. The broader dependency type-check limitation is recorded separately in
the integration ledger.
