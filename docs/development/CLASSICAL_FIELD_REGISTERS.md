# Classical field registers in component graphs

Simulation owns `classical_field_registers.py`, adapting finite-field assignment
and array collection from unchanged Qualtran v0.7.0 `simulation/classical_sim.py`
at `bcb17bb25360ed68ce3bc084645db3e929799389`. It reuses the existing
`FiniteFieldEncoding` numerical owner. Six original Git blobs match the frozen
inventory: classical simulation and its tests, data types, registers, composite
graphs and GF2Addition. Attribution remains in the native files; workspace
source/resource evidence is `g2-classical-field-registers-source.json`.

## Ownership and interfaces

The existing `classical_registers.py` now owns shared binding and collection
traversal. Its integer entry points keep their signatures and delegate to that
shared implementation; the former integer-only traversal is removed. The field
entry points supply only their domain validators and storage allocation.
Substituted shared providers prove both consumers use this boundary.

`bind_field_register_values(name, values, encoding, shape=...)` performs named
lookup, concrete shape validation, scalar Expr bypass, optional stochastic
resolution and domain validation in source order. It does not coerce scalar
integers to field elements. `FiniteFieldEncoding.assert_valid_val` retains the
nominal field-class requirement; array validation retains the source's
range-only policy. `asanyarray` preserves field-array subclasses and row-major
indices. Plain numeric arrays may pass binding but subsequently fail field
collection; that source failure remains visible. Sampled symbolic values do not
re-enter the earlier Expr bypass.

`collect_field_register_value` returns scalar objects unchanged. For arrays,
it allocates `encoding.gf_type.Zeros(shape)` before indexed writes, including
empty arrays, preserving the source field class, conversion errors and missing
assignment errors. It does not substitute integer array storage. Field class
resolution remains lazy on scalar collection.

`ComponentClassicalState` accepts an optional explicit mapping from Core `RegisterType`
to Simulation `FiniteFieldEncoding` through `field_encodings`. The mapping is
copied, then used only for declared field ports; its projected descriptor must
match exactly before value callbacks run. This override path does not reconstruct
metadata. Missing encoders now use the explicit
[binary metadata resolver](BINARY_FIELD_ENCODING.md). The class consumes the same Core SSA values,
Compiler-supplied operation positions and existing action/distribution/phase
owners. Simulation imports no Compiler implementation and owns no graph rewrite.

Core already admits resolved binary quantum fields. This change adds numerical
execution for that admitted subset; it does not expand Core admission to odd
characteristics, signed words or classical-only field metadata. Standalone field
binding/collection can still use the existing wider numerical field codecs.
Physical word binding remains MSB-first. The numerical owner allocates no
physical workspace; existing emitters retain work ownership. This is a
`single_device_fast_path` inspection path, not a new backend or protected API.

A typed numerical identity uses the explicit field boundary:

```python
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

field = FiniteFieldEncoding(2, 3)
dtype = field.register_type()
signature = Signature([Register("x", dtype, (2,))])
leaf_builder = ComponentProgramBuilder(signature, name="identity")
leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
builder = ComponentProgramBuilder(signature, (leaf,))
record = builder.finalize(builder.call_dict("identity", builder.inputs))
value = field.gf_type([1, 7])
state = ComponentClassicalState(
    record, (0,), {"x": value}, lambda function, values: values,
    field_encodings={dtype: field},
)
result = state.simulate()["x"]
assert isinstance(result, field.gf_type)
assert (result == value).all()
```

## Verification

- 40 new tests compare genuine source input assignment and `_get_in_vals` for
  scalar/vector/matrix/empty arrays across GF(2), GF(4), GF(8) and GF(9). They
  retain field classes, scalar identity, row-major order and caller inputs.
- Negative cases cover plain scalar integers, other field classes, missing
  values/indices, wrong shapes, out-of-range arrays, floating/NaN arrays and
  ordinary array conversion. Expr bypass and one-shot distribution validation
  remain on the same shared path as integer binding.
- Typed identity graphs consume real field scalars/arrays. GF(2), GF(4) and GF(8)
  add graphs exhaust all 4, 16 and 64 inputs respectively through the native
  state and unchanged source graph evaluation. Reconstructed full complex
  matrices match source tensor contraction.
- Each physical add stream matches original decomposition gate by gate and by
  operand, using source CNOT decompositions. A complex superposition runs through
  ordinary `circuit.append(op)` and matches the full matrix; a second append
  restores it. The unused clean spectator remains zero. These circuits have no
  measurements, so measurement-branch checks are not applicable to this new
  consumer; existing measured integer/phase consumers remain in regression.
- Concentrated regression: 2,757 passed, 207 recorded source/native numerical
  warnings. Focused binding/graph regression: 198 passed without warnings.
  New tests alone: 40 passed without warnings.
- Coverage: field owner 100% (eight statements); shared register owner 96%
  (55 statements, 22 branches); graph state 98% (118 statements, 48 branches).
  Total measured scope is 97%. Remaining uncovered paths are optional SymPy
  absence, a storage-loop fallthrough and the two typed graph refusal sites.
- Strict typing of four files, Black, Ruff, Simulation scope, architecture,
  multi-level IR and protected API checks pass. CPU/representation/TN regression:
  84 passed, two known optional dependency/deprecation warnings.

- Default smoke/unit run: 7 failed, 21,890 passed, 300 skipped, 1,037 warnings in
  118.89 seconds, stopped at its maxfail bound. All seven failure IDs match the
  existing baseline; no new failure was observed. This is not a fully green suite.

## Separate resources and remaining boundaries

The numerical helpers and graph admission emit zero primitive gates, own zero
physical work bits and use zero feedback slots. Numerical assignment/array
storage is separate from physical qubit storage.

| Physical consumer | Measured primitives | Original QEC model | Physical work / measurement / feedback |
| --- | --- | --- | --- |
| GF(2) addition | CX1 | Clifford1 | 0 / 0 / 0 |
| GF(4) addition | CX2 | Clifford2 | 0 / 0 / 0 |
| GF(8) addition | CX3 | Clifford3 | 0 / 0 / 0 |

Each test adds one unused clean spectator. Primitive equality is verified
independently of the model categories; there is no new logical-T, GPU, hardware
or scalability claim. Quantum users continue `circuit.append(op)`.

Q-0882 and Q-0883 remain open. Source-wide dtype dispatch, foreign lazy
decomposition exception conversion and complete original test-file replacement
remain incomplete. [Binary metadata resolution](BINARY_FIELD_ENCODING.md) now restores missing
encoders automatically for already admitted binary quantum fields; explicit
matching encoders remain optional overrides. This native representation adapter
is separate from an upstream QGF constructor.
The source's empty external array finalization still raises
`KeyError(RightDangle)` while native typed return metadata collects an empty
field array. Tests preserve and report that difference. Wider standalone field
domains do not imply wider Core executable admission, and field array codecs'
earlier vectorizer limits remain separate. No new child ID or completion
denominator is introduced. G3 remains unstarted.
