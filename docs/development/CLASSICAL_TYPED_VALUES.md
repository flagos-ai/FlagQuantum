# Typed classical value binding and collection

Simulation owns `bind_typed_register_values(register, values)` and
`collect_typed_register_value(register, assignments)`. These functions connect
concrete native declarations to existing numerical validators and shared binding
and collection traversal. They do not admit new executable types, lower gates,
discover foreign Python methods or create a second interpreter.

`ComponentClassicalState` now consumes these two functions instead of maintaining
its own bit/integer/opaque/field dispatch. The constructor's Core verification,
binary-field admission, override metadata matching, call order, occurrence IDs,
phase policy and RNG policy remain with their existing owners. Callers continue
to write `circuit.append(op)` for physical execution.

```python
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.classical_typed_values import (
    bind_typed_register_values,
    collect_typed_register_value,
)

register = Register("angle", fixed_point_register_type(3, 2), shape=(2,))
assignments = bind_typed_register_values(register, {"angle": [1, 3]})
values = collect_typed_register_value(register, assignments)
```

Fixed point uses stored integers, not rescaled reals. Bounded integers validate
against their bound; Montgomery stored values validate against their width,
independently of its domain enumeration. Ones complement keeps the source
validator's inclusive endpoints and its distinct encoding/domain behavior.
Field scalar validation is nominal; array validation retains source range-only
behavior. Scalar symbolic bypass and one-time distribution resolution stay in
`classical_registers`; a sampled symbolic object does not acquire a second bypass.

The source `_numpy_dtype_from_qlt_dtype` special-cases quantum QUInt/QAny/QInt,
plus both bit declarations. Other arrays, including bounded, fixed-point,
Montgomery, ones complement and classical integer declarations, use object
storage. `_empty_ndarray_from_reg` specializes QGF only: quantum field arrays
retain their field class, while CGF arrays use object storage. Scalar collection
returns the original object. NaN/integer conversion errors and warnings remain;
no casting is added to make a source failure pass.

The numerical adapter requires concrete metadata. Executable Program admission
remains separate; this is not source-wide graph execution. Explicit resolved
field overrides remain available, and their caller owns metadata matching.
Source diagnostic wording uses foreign register context; native validation uses
its existing numerical messages. The locked source is unchanged.

## Verification and resources

Tests compare real `ClassicalSimState._update_assign_from_vals` and `_get_in_vals`
against native binding and collection. They cover 17 word declarations and 16
field declarations across scalar, vector, matrix and empty shapes, raw floats,
NaN, scalar identity, array storage classes and native integers. Symbolic and
sampled-symbolic cases retain validation order. A field override replacement
proves default resolution is bypassed. Existing native state consumers exercise
the replaced binding path.

The measured Add consumer checks all 64 inputs, a complete complex matrix and all
16 measurement branches. Both work bits return to zero. Its primitive stream is
CX40, H12, T8, Tdg8, S4, measurement4, CZ4 and reset4. The separate source QEC
model reports AND4, Clifford22 and measurement4. Numerical binding itself emits
zero gates and allocates no quantum work bits. No physical decomposition is
inferred for wider numerical declarations.

Local verification: 151 new tests, 3707 concentrated regressions, and 84 CPU/TN checks passed. Strict typing of four changed Python files, Black, Ruff, Simulation scope, architecture, internal IR and protected API checks passed. The typed owner reached 100% statement/branch coverage (53 statements, 22 branches); the existing state owner reached 97% (132/46). Seven pinned source blobs matched the inventory.

Default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings in 145.45s (0:02:25)`. All observed failure IDs belong to the seven known baseline cases; 0 baseline cases were not observed before the configured stop. This is not a green full suite. Evidence is retained in `ftqc-reference-env/g2-classical-typed-values-default-comparison.json`.

## Subsequent symbolic composition

[Symbolic metadata](CLASSICAL_SYMBOLIC_METADATA.md) replaces the concrete-only numerical adapter described above. Existing numerical owners now receive symbolic widths/value parameters; bit codecs still require concrete metadata. Source operation-specific successes and failures, rather than blanket symbolic support, are verified.
