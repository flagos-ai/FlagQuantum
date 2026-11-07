# QROM table metadata

`compiler/registers/qrom_table.py` owns concrete data normalization, selection
axes, target array shapes, and default widths. It ports QROMBase's data and
concrete/symbolic shape factories from Qualtran v0.7.0, commit
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

Symbolic shape metadata and unknown-data resource queries are supported through
`from_shape` and `resource_formula`. Source register objects, classical evaluation
and LEFT-side target erasure remain unmapped here.
The source data constructor's ineffective width guard can retain undersized or
negative values; native metadata retains those values too, while the existing
unsigned IR/model validators reject them. Replacement uses the source's maximum
bit-length check. Invalid target suffixes and mismatched selection shapes are
rejected rather than reproducing the source empty-shape sentinel ambiguity.
Zero-width targets and nonuniform lookups with omitted singleton axes remain
subject to the existing emitter's documented rejection. None of these boundaries
is claimed as complete QROMBase parity.


## Unknown-data symbolic queries (G2-ROM10)

`QromTable.from_shape` retains symbolic dimensions, explicit selection widths,
controls and target array axes using the existing `SymbolicInt` and `bit_length`
authorities. `resource_formula(legacy_shims=...)` delegates to
`source_qrom_resource_formula`. It is a value-independent unknown-data query,
including when called on a loaded table. `resource_model` instead requires data
and uses data-dependent pruning. The two queries are deliberately distinct.

Run the example with `--symbolic` to print the logical formula and T equivalents
without allocating a table or emitting gates. `test_qrom_table_symbolic.py`
compares factory fields and all seven QEC categories plus peak width with the
unmodified original source in both profiles. It includes symbolic controls,
multiple axes, grouped targets, explicit address widths, omitted singleton axes,
zero-width targets, symbolic integer literals, and zero-rank/single-row metadata.
The original single-row unknown-data declaration returns -1 And and -1
measurements; this diagnostic is retained, not clamped into a physical estimate.

Dimensions must be explicitly resolved to Python integers for physical binding.
No symbolic dimension is inferred from replacement data, and Sympy integer
literals are not silently cast. Replacement can retain an exactly matching
literal symbolic shape as the original does; physical binding still rejects
symbolic dimensions. Two resolved nonuniform tables compare source gate streams
and every measurement history with coherent initial targets and inactive controls,
absolute complex phase and zero work. A replacement test proves the table
consumes the shared formula instead of reimplementing it. Concrete QROM/QROAM
regressions continue to run through the previously accepted emitters.

## Named registers and packed binding (G2-ROM11)

`register_layout` maps QROMBase's control/selection/target signature into the
existing FQ `(name, width)` partition layout. Target array axes are flattened
only for physical port packing; their shape remains in the table metadata.
One surviving selection axis is named `selection`; multiple surviving axes are
numbered in source signature order. Concrete singleton axes and zero-width
targets are omitted, while symbolic literals follow the original symbolic
inclusion rules. The original bounded-selection dtype failure is also rejected
before offering a physical layout.

`bind_packed(ports, workspace, classical_bits)` delegates partition validation to
`partition_register`, reconstructs grouped target words, and calls `bind`.
The example now uses this path before `circuit.append(op)`. No gates are added
by partitioning. All 11 original gate/channel cases also compare packed binding
with the explicit-port binding, including multi-axis tables, grouped targets,
controls, omitted singleton axes and zero selection. Twenty-five new tests
compare source signature names and total widths, symbolic controls/literals,
invalid packing and work/feedback ownership. Source dtype objects and
LEFT/RIGHT ownership semantics are not introduced as new FQ public concepts.
