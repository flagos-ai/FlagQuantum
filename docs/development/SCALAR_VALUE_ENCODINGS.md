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
