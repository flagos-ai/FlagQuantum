# Concrete QROM table metadata

`compiler/registers/qrom_table.py` owns concrete data normalization, selection
axes, target array shapes, and default widths. It ports QROMBase's data and
concrete shape factories from Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Copyright and source attribution
remain in the implementation. Production code never imports Qualtran.

`QromTable.from_data` converts values to native integers in row-major order and
infers each width from the original maximum, with a minimum of one bit. Target
array axes follow selection axes. `from_shape(...).with_data(...)` validates
replacement shapes and widths without mutating the shape-only object. The
existing `(word count, word width)` layout is shared with the concrete QROM
resource model. Physical bits remain big endian.

`bind` delegates to `source_qrom_ir`: traversal, measured And cleanup, and gate
order have one existing authority. The caller supplies all ports, clean work
and classical slots. `resource_model` delegates to `source_qrom_cost` without
emitting a circuit. It preserves the modern and legacy source cost profiles.
Numerical quantum execution remains outside Compiler.

Run `python -m examples.algorithms.register_qrom_table` for `circuit.append(op)`
with a shape-only table bound to data, complex amplitudes, clean work and
separate model/primitive output. `tests/unit/test_source_qrom.py` now routes its
11 source gate/channel cases through this metadata binding as well as comparing
the raw emitter, preserving full complex channels and exhaustive small raw
measurement histories. `test_qrom_table.py` checks original factory metadata,
float conversion, multiple datasets, grouped target axes, replacement and
shape-only failures.

## Explicit boundaries

This slice supports concrete dimensions. Symbolic shapes, source register
objects, classical evaluation and LEFT-side target erasure remain unmapped here.
The source data constructor's ineffective width guard can retain undersized or
negative values; native metadata retains those values too, while the existing
unsigned IR/model validators reject them. Replacement uses the source's maximum
bit-length check. Invalid target suffixes and mismatched selection shapes are
rejected rather than reproducing the source empty-shape sentinel ambiguity.
Zero-width targets and nonuniform lookups with omitted singleton axes remain
subject to the existing emitter's documented rejection. None of these boundaries
is claimed as complete QROMBase parity.
