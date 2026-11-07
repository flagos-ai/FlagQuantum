# Integer value encodings

Simulation owns numeric integer encoding and array conversion in
`integer_encoding.py`. Compiler's `register_bits.py` remains the authority for
its stricter concrete compile-time constants and physical register binding; a
numerical array codec does not replace those validation rules. Neither domain
imports the other and no Core register/dtype contract is added.

The source is unmodified Qualtran v0.7.0 at
bcb17bb25360ed68ce3bc084645db3e929799389: BitEncoding, _UInt and _Int in
`qualtran/_infra/data_types.py`. Unsigned scalar codecs also serve the stored
Montgomery word encoding used by SingleWindowModMul. Montgomery array methods
use the source's generic vectorization and are not claimed to be the _UInt
optimized array path delivered here.

## Interfaces

- `integer_to_bits(value, width, signed=False)` returns a big-endian Python list.
- `integer_from_bits(bits, width, signed=False)` returns the source scalar integer.
- `integer_array_to_bits(values, width, signed=False)` returns the source uint8
  array, preserving its actual shape and failure behavior.
- `integer_array_from_bits(bits, width, signed=False)` preserves the original
  result dtype and vectorized inference; it does not force all results to uint64.

Scalar encoding reuses `integer_values.py`. Its bounds now evaluate using the
source arithmetic on width instead of adding a native width-constructor check.
This retains NumPy-width overflow and raw numerical diagnostic widths; it does
not make such values valid physical register dimensions. Existing binding,
output collection and typed negation consumers are included in regression.

Unsigned scalar encoding validates the value and uses the original integer
formatting. Signed encoding rejects symbolic widths (including Sympy integer
literals), validates the value and calls NumPy binary_repr. Scalar unsigned
decoding ignores declared width; signed decoding uses first-bit sign and suffix
complement, including the width-one special case. No new length or binary-element
check is inserted into these diagnostic methods. In particular, unsigned scalar
zero-width encoding yields `[0]` and a one-bit signed decode ignores the suffix,
as in the source. These are not new quantum word guarantees.

The <=64-bit unsigned array encoder retains the original unsigned conversion,
mask broadcasting and transpose. It does not add scalar range checks, so an
out-of-width element may be truncated. Incompatible multidimensional shapes,
noninteger dtypes and empty vectorization failures are preserved. Signed arrays
use generic per-element vectorization. Above 64 bits, both source paths retain
vectorized dtype inference, including overflow when the first result selects a
narrow dtype. The unsigned array decoder retains its exact width comparison,
uint64 basis and sum; malformed nonbinary arrays are not silently validated.

## Acceptance and use

G2-VAL12: 389 codec cases plus 12 additional source width-validation cases.
The tests compare values, result types, array dtypes/shapes and original exception
classes for concrete, zero, negative diagnostic, NumPy and symbolic widths.
Existing source warnings remain visible. Source references are not patched.

A minimal numerical example uses only native interfaces:

```python
import numpy as np
from flagquantum.simulation.integer_encoding import integer_to_bits, integer_from_bits, integer_array_to_bits
assert integer_to_bits(-2, 3, signed=True) == [1, 1, 0]
assert integer_from_bits([1, 1, 0], 3, signed=True) == -2
np.testing.assert_array_equal(integer_array_to_bits(np.array([1, 2]), 2), [[0, 1], [1, 0]])
```

The second commit in this batch connects these codecs to the bit-array Montgomery
window and its history chain. The shared scalar codec does not allocate quantum
work or add gates.
The unchanged source graph remains the reference for quantum amplitudes,
measurement histories and clean-work evidence, including known source graph/value
disagreements. [Ones' complement value codecs](SCALAR_VALUE_ENCODINGS.md) and
[fixed-point stored integer codecs](FIXED_POINT_VALUE_ENCODING.md) are now separate
accepted slices. Real fixed-point quantization and finite-field dtype codecs
remain unaccepted portions of the source inventory.

## Shared default array vectorization (G2-VAL14)

`vectorized_integer_array_to_bits` and `vectorized_integer_array_from_bits` factor
the exact original BitEncoding default into reusable numerical functions. The
signed and >64-bit unsigned routes consume them; the latter retains its original
width precheck before dispatch. Stored Montgomery arrays use the same default
without an unsigned-mask precheck or forced output dtype. Per-element scalar
codec evaluation, vectorize signatures and inference are unchanged.

These interfaces preserve the source's differences: a two-dimensional stored
Montgomery array can encode to input_shape+(width,), while the optimized _UInt
mask path may fail broadcasting. Default array decoding does not add _UInt's
width equality guard. Sympy integer widths may work in default unsigned encoding
while the optimized unsigned array path rejects symbolic width before evaluating
scalars. Empty vectorization and arbitrary-wide output inference failures remain.

There are 140 new source Montgomery array and replacement tests. Existing signed
and wide unsigned consumers are tested with an injected helper implementation,
proving they use this shared route. No reference implementation is replaced or
modified, and the earlier 389 scalar/array codec tests remain in regression.
