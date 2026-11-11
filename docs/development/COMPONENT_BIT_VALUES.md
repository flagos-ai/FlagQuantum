# Bit validation in classical component graphs

Simulation's existing classical SSA state now dispatches quantum and classical
bit ports to the existing `bit_encoding.validate_bit_value` and
`validate_bit_array` through shared register binding. The former integer-only
validation path is removed for these ports. Collection still uses the existing
uint8 array storage; scalar objects remain unchanged. No new validator, dtype,
IR, codec, registry, backend or protected API is introduced.

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`: `_Bit.assert_valid_val`,
`assert_valid_val_array` and `ClassicalSimState._update_assign_from_vals`.
Six original Git blobs match the frozen inventory in workspace
`g2-component-bit-values-source.json`, including measurement and graph sources.
The reference is not edited. The numerical bit owner already retains source
attribution; this batch connects its existing semantics to the graph consumer.

## Corrected counterexamples

| Input to a bit identity graph | Before this batch | Unchanged source and corrected native behavior |
| --- | --- | --- |
| Scalar `1.0` | Integer-type rejection | Accepted, same float object returned |
| Array `[0.5]` | Accepted and truncated to uint8 zero | Rejected before storage conversion |
| Array `[NaN]` | Accepted and cast to zero with a warning | Rejected before storage conversion |

The before-state evidence loads the exact previous graph-state Git blob from
`e611b8c7be73265a7d020df98b0d62a06ea7ecaa` under a temporary test module name,
then compares source and corrected behavior. It does not edit an upstream file
or alter the production import path.

## Ownership and ordering

Bit validation uses equality to zero or one rather than Python/NumPy integer
membership or a width-one interval. Floating, complex, boolean and other numeric
objects equal to zero/one retain source acceptance. The source Expr bypass
still precedes validation, and a sampled Expr does not re-enter that bypass.

Array equality is checked elementwise before row-major collection. Valid arrays
retain source uint8 storage and conversion warning categories; fractional and
NaN entries fail without a cast warning. Shared binding still owns named lookup,
shape validation, symbolic handling and one-shot distribution resolution. The
existing phase owner still observes outputs only after successful assignment.
A width-one unsigned register retains its integer policy; it is not a bit port.

This is a `single_device_fast_path` numerical fix. Compiler supplies graph order
and owns transforms/emission. Physical bit order, gate streams, measurement
feedback and workspace ownership are unchanged. Users continue
`circuit.append(op)` for quantum execution.

A numerical bit identity can retain a valid float representation:

```python
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.component_classical_state import ComponentClassicalState

signature = Signature.build(x=1)
leaf_builder = ComponentProgramBuilder(signature, name="identity")
leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
builder = ComponentProgramBuilder(signature, (leaf,))
record = builder.finalize(builder.call_dict("identity", builder.inputs))
value = 1.0
state = ComponentClassicalState(record, (0,), {"x": value}, lambda f, v: v)
assert state.simulate()["x"] is value
```

## Verification

- 97 new cases compare genuine unchanged source state execution with native
  execution for QBit and CBit scalars and arrays. They cover floats, complex
  values, NumPy scalar/array objects, booleans, fractions, decimals, symbolic
  values, invalid ranges, fractions/NaNs in arrays and empty arrays.
- Accepted scalar object identity and array uint8 dtype, warnings and input
  immutability match source. Width-one unsigned rejection remains intact.
  Output assignment precedes pure/phased rejection as in the original state.
- Fixed floating/NumPy-floating/complex measurement outcomes retain their exact
  object and phase. All four input/outcome pairs are compared for each of three
  representations: 12 physical branch evaluations, with unnormalized complex
  amplitude phase/sqrt(2) and a zero clean spectator.
- A substituted shared binder proves that graph bit ports use the existing bit
  validators and bypass integer validation. Existing coherent full-matrix,
  append, original gate/operand/feedback and measured inverse histories remain
  in concentrated regression.
- Concentrated regression: 2,898 passed, 207 recorded source/native numerical
  warnings. New tests alone: 97 passed without warnings. Coverage regression:
  221 passed without warnings. The touched graph owner covers 97% of 124
  statements and 52 branches; three graph refusal sites remain uncovered.
- Strict typing of two files, Black, Ruff, Simulation scope, architecture,
  multi-level IR and protected API checks pass. CPU/representation/TN regression:
  84 passed, two known optional-dependency/deprecation warnings.

- Default smoke/unit run: 7 failed, 21,890 passed, 300 skipped, 1,037 warnings in
  141.08 seconds, stopped at its maxfail bound. All seven observed failure IDs
  match the existing baseline; no new failure was observed. This is not a fully
  green default suite.

## Separate resources and remaining boundaries

The numerical validation change emits zero gates and allocates zero physical
work bits or feedback slots. Its identity consumers emit no gates. The existing
X-measurement physical consumer emits H1 and measurement1, with one unused clean
spectator and one measurement record. The original QEC model reports measurement1
and omits the physical H. Measured primitives and model categories remain
separate; no gate count equality is claimed between those accounts.

Native error messages use the numerical validator's `val` context rather than
foreign graph instance names such as `LeftDangle.x`; this diagnostic-text boundary
is explicit. The source empty external array still raises `KeyError(RightDangle)`
while native typed return metadata can collect an empty array. Q-0882 and Q-0883
remain open for source-wide dtype/decomposition and original-test replacement.
No new child ID or completion denominator is added. G3 remains unstarted.
