# Binary field metadata resolution

Simulation owns `binary_field_encoding.resolve_binary_field_encoding`, the
inverse of the existing `FiniteFieldEncoding.register_type` projection for
Core-admitted binary quantum fields. Core remains the owner of neutral metadata
and executable admission. Numerical galois objects remain inside Simulation;
there is no new dtype, graph, registry, global cache or backend selection.
This is a `single_device_fast_path` numerical inspection change.

The pinned Qualtran v0.7.0 source at
`bcb17bb25360ed68ce3bc084645db3e929799389` stores a QGF object directly in each
register. It therefore has no corresponding inverse-metadata resolver. This is
an explicit native representation adapter; the existing field constructor,
validation and codecs still follow the unchanged source `_GF` call chain.
Six original source blobs match the frozen inventory in workspace
`g2-binary-field-resolution-source.json`. The reference files are not edited.

## Resolution and boundaries

1. Run the existing Core `executable_register_width` admission before optional
   library import. Require a finite-field descriptor; reject wider standalone
   field declarations, symbolic parameters and noncanonical metadata.
2. If polynomial metadata is absent, pass explicit `None` to the existing
   `FiniteFieldEncoding` constructor. Do not replace absence with its omitted
   default-polynomial behavior.
3. Otherwise reconstruct the polynomial's coefficient field from its stored
   characteristic, degree, primitive element and modulus, then use the stored
   polynomial integer. No coefficient information is discarded. The target
   degree-one case retains even the polynomial ignored by the original codec.
4. Project the restored declaration back to metadata and require exact equality.
   Inconsistent library parameters propagate their error; no integer coercion or
   repair is applied. Target `gf_type` property evaluation remains lazy.

`ComponentClassicalState` resolves a missing field encoder once per descriptor
within that state and reuses it across declared ports and calls. Optional explicit
`field_encodings` still take precedence and must match the same descriptor.
Integer graphs do not request a field encoder. There is no cross-state cache.
Pure/phased action ordering, RNG/fixed-value policy, assignments and physical
emission remain with their existing owners. This does not import Compiler into
Simulation or extend the protected public API.

A numerical graph can now omit its encoding map:

```python
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.binary_field_encoding import resolve_binary_field_encoding
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

dtype = FiniteFieldEncoding(2, 3).register_type()
field = resolve_binary_field_encoding(dtype)
signature = Signature([Register("x", dtype, (2,))])
leaf_builder = ComponentProgramBuilder(signature, name="identity")
leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
builder = ComponentProgramBuilder(signature, (leaf,))
record = builder.finalize(builder.call_dict("identity", builder.inputs))
value = field.gf_type([1, 7])
state = ComponentClassicalState(record, (0,), {"x": value}, lambda f, v: v)
output = state.simulate()["x"]
assert isinstance(output, field.gf_type)
assert (output == value).all()
```

## Verification

- 44 new tests cover widths 1, 2, 3, 4 and 8, absent/default polynomials, complete
  scalar codec domains and exact source field-class identity. The GF(8) moduli
  11 and 13 preserve distinct source multiplication tables and results.
- Six coefficient fields include prime-field primitive elements 3 versus 5 and
  extension-field moduli 11 versus 13. Even a degree-one target retains those
  polynomial identities instead of collapsing them to its resolved GF(2) class.
- Admission failures precede optional import. Malformed order, width, shape of
  parameter declarations, duplicates, reordered/partial fields, symbolic and
  classical descriptors are rejected. Inconsistent coefficient metadata is not
  repaired. Explicit override precedence, one resolution per descriptor and no
  resolution for integer graphs are verified through substituted providers.
- Six automatic graph cases exhaust 4, 16 and four sets of 64 input pairs:
  GF(2), GF(4), and GF(8) with default, absent, 11 and 13 polynomial metadata.
  Every table equals the original source table; full complex matrices equal
  source tensor contraction. Typed field-array passthrough and the retained
  source empty-array finalization difference are tested separately.
- Each emitted Add matches original decomposition gates and operands. A complete
  complex superposition matches the matrix through `circuit.append(op)`; a second
  append restores it and the unused clean spectator stays zero. These consumers
  contain no measurements; no fictitious measurement branches are attributed to
  them. Existing measured integer/phase consumers remain in regression.
- Concentrated regression: 2,801 passed, 207 recorded source/native numerical
  warnings. Coverage regression: 124 passed without warnings. Resolution owner:
  23 statements/eight branches, 100%; graph state: 121 statements/50 branches,
  96%; combined measured scope 97%. Three remaining graph refusal sites are
  uncovered. This is not whole-repository coverage.
- Strict typing of four files, Black, Ruff, scope, architecture, multi-level IR
  and protected API checks pass. CPU/representation/TN: 84 passed, two known
  optional-dependency/deprecation warnings.

- Default smoke/unit run: 5 failed, 21,783 passed, 300 skipped, 1,037 warnings in
  118.07 seconds, stopped at its maxfail bound. The five observed failure IDs
  belong to the seven-ID baseline; two baseline failures were not observed. No
  new failure was observed. This is not a fully green or complete default suite.

## Separate resources and remaining work

Resolution and classical execution emit zero primitive gates, own zero physical
work bits and use zero feedback slots. Numerical objects and assignments remain
separate from physical resources.

| Physical consumer | Measured primitives | Original QEC model | Physical work / measurements / feedback |
| --- | --- | --- | --- |
| GF(2) addition | CX1 | Clifford1 | 0 / 0 / 0 |
| GF(4) addition | CX2 | Clifford2 | 0 / 0 / 0 |
| Each GF(8) addition | CX3 | Clifford3 | 0 / 0 / 0 |

Each physical test includes one unused clean spectator. Gate-stream equality is
checked separately from source model categories. No GPU, hardware or distributed
capacity claim follows from these CPU inspections.

Q-0882 and Q-0883 remain open. Automatic resolution covers already admitted binary
quantum fields, not all source dtypes. Signed/odd-characteristic/classical field
Core admission is unchanged; wider standalone codecs retain their separate
scope. Foreign decomposition exception translation and complete original test
replacement remain open. The source empty external array still fails with
`KeyError(RightDangle)` while native typed metadata can collect its empty return.
No new child ID or parent denominator is introduced. G3 remains unstarted.
