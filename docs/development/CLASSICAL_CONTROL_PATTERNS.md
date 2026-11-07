# Classical control-pattern grouping

Simulation owns `classical_control_patterns.normalize_control_patterns`. It
ports the original `_cvs_convert` branch order from Qualtran v0.7.0
`_infra/controlled.py` at `bcb17bb25360ed68ce3bc084645db3e929799389`.
Native unknown arrays use the accepted `SymbolicArrayShape`; foreign types do
not cross the production boundary. The result feeds existing activation and
controlled-value callbacks, not a parallel control-specification hierarchy.

| Input | Bound pattern groups |
|---|---|
| Integer or NumPy integer | One scalar array |
| NumPy array, including a subclass | One group; retain array identity |
| Integer sequence, including an empty sequence | One array group |
| Nested sequence | One group per outer item |
| Native symbolic array | One group; retain descriptor identity |
| Sequence containing symbolic arrays | Retain those descriptors among array groups |

An ndarray of shape `(2, 2)` is one matrix register; the corresponding nested
list is two vector registers. Python booleans follow the integer path; NumPy
boolean scalars retain the source iterable failure, and a list of NumPy booleans
retains its separate scalar groups. No domain, width, broadcasting or register
count validation is added. Unknown sequence lengths and arbitrary symbolic
objects are not treated as symbolic arrays.

The input annotation follows the original recursive sequence contract. Runtime
single-pass iterables retain the original diagnostic behavior: `all` consumes
items, the first failing item is consumed, and converting an exhausted all-integer
iterator produces a scalar object array. This is not silently repaired into a
different grouping algorithm.

Run `python -m examples.algorithms.numerical_value_dimensions`. Both its
classical oracle and `circuit.append(op)` path consume the shared converter.
The existing emitter still owns bit order, work allocation, measured inverse
and gate validation. A replacement test changes conversion without changing
the consumer and checks the new stream against the corresponding original
controlled-X source stream.

`tests/unit/test_classical_control_patterns.py` adds 48 direct-source checks for
grouping, dtype, aliases, shape identity, diagnostic failures and iterator order.
Together with the dimensions, activation, controlled-value and symbolic-value
tests, 520 cases pass without skips or warnings. The integrated matrix control
retains exact primitive parity, complete complex amplitudes, all 8 forward and
64 round-trip measurement histories and clean work, with both QEC profiles.
Model counts and emitted instructions remain separately reported as documented
in [numerical dimensions](NUMERICAL_VALUE_DIMENSIONS.md).
