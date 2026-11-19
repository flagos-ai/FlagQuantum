# Fixed-point real conversion boundary

## Ownership and interface

The complete `data_types.py` source audit inventories 224 definition occurrences
and its original test file inventories 52. This dependency belongs to Q-0160 and
Q-0161. The complete parent audit is recorded in DATA_TYPE_SOURCE_FILE_AUDIT.md.

Ecosystem owns `fixed_point.py`, the optional external fixed-point object
boundary. Functions take total width, fractional width and signedness. They
convert real inputs to big-endian bits or the existing raw integer representation,
construct external templates, validate exact representability, decode external
values and iterate the real domain. Simulation owns the existing integer codecs,
raw fixed-point domains and shape validation. Core descriptors remain neutral.
External objects stop at Ecosystem. The path is `single_device_fast_path`.

The implementation follows the unchanged `_Fxp` call chain in Qualtran v0.7.0
revision `bcb17bb25360ed68ce3bc084645db3e929799389`. Template options, conversion
order, wrapping, sign-magnitude handling, exactness checks and bit-string slicing
are preserved. The native diagnostic names its own fixed-point boundary.
QFxp/CFxp forwarding continues to use the same common numerical implementation.
There are no allocations, measurements or quantum instructions in this adapter.
Actual circuit consumers retain their existing decomposition and resource owners.

## Optional dependency review

The reviewed reference is fxpmath 0.4.10, MIT licensed, depending on NumPy. The
installed distribution's license and metadata are retained in the isolated
reference environment. It is imported lazily at this explicitly invoked
Ecosystem boundary. Dependency manifests and the default installation are unchanged.
Reference parity requires the locked environment and records the installed
version. Other versions have no parity certification from this batch.

Exact reproduction of the source's external template and conversion behavior
requires its original external implementation. Inputs and output bit lists or
integers form the replacement interface. A replacement must preserve all source
conversion and mutation semantics, pass the same conformance suite, and remove
these external-object entry points only through any then-applicable API process.
The adapter owns the optional dependency and its exit. Core, Compiler, Runtime
and Simulation gain no fxpmath imports. Missing installation raises the original
import failure when the external conversion is requested.

## Complete parent audit

The 224-definition ownership map and all 285 unchanged original test cases are
recorded in DATA_TYPE_SOURCE_FILE_AUDIT.md. Legacy custom scalar callbacks, opaque
validation/domain behavior and polynomial symbolic metadata use existing native
owners. Wider classical Program execution remains governed by the separately
pending Proposal 108.

## Executable conversion example

Run with the optional fxpmath installation in the reference environment.

```python
from flagquantum.ecosystem.fixed_point import (
    fixed_point_bits_to_real,
    fixed_point_real_to_bits,
    fixed_point_real_to_integer,
)

assert fixed_point_real_to_integer(-1.5, 6, 4, signed=True) == -24
bits = fixed_point_real_to_bits(-0.625, 4, 3, signed=True)
assert bits == [1, 0, 1, 1]
assert fixed_point_bits_to_real(bits, 4, 3, signed=True).get_val() == -0.625
```

The existing `examples.algorithms.classical_fixed_point_values` demonstrates the
raw integer, phase-gradient append and inverse consumer. The conversion tests
add quantization into that actual consumer and compare every small measurement
history with the frozen source.
