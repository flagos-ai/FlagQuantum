# Indexed component dispatch

G2-IDX01 owns the shared register dispatch underlying locked Qualtran v0.7.0
ApplyLthBloq and ApplyGateToLthQubit. Compiler binds ordered big-endian address
axes, concrete extents, positive quantum control wires, explicit clean work
and result slots. The callback accepts a C-order flattened index and one
positive predicate wire, and returns controlled child instructions. It must
preserve that predicate and all address/control words, and clean its own work.

The parent owns only address/predicate work. Child storage remains caller-owned
and must be declared separately by its operation. Nested address axes retain
separate bounded traversals; flattening is only used for dispatch. Reuse existing
MultiAnd, bounded traversal and measured uncompute instead of a second decoder.
All parent clean wires enter/leave zero; original inputs are preserved.

The target-position adapter supplies one Pauli/I label per flattened address
and one target wire per entry. The source ravel_multi_index order is retained.
Source padded-address behavior is preserved, not silently changed to identity.
The positive control interface does not reinterpret source control_val=0.
Any empty address axis and an uncontrolled singleton first axis are refused
before emission because the locked source does not define an executable path.
No Core opcode, public export, simulator or upstream dependency is introduced.

References: Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389,
`bloqs/multiplexers/apply_lth_bloq.py`,
`apply_gate_to_lth_target.py`, and `unary_iteration_bloq.py`.

## Local evidence and resource bases

Forty-two source configurations compare complete ordered gate dependencies,
measurement feedback, T/T-dagger count, measurement count and peak used width
against unmodified pinned source. They include multidimensional extents,
oversized address words, up to three positive controls, singleton populated
axes with nonempty words, and all I/X/Y/Z leaf families. The reserved parent
work is sum(axis widths)+control count-1. Source peak width counts only work
actually used; declared ports and the reserved clean buffer remain separate.

Twenty-four complete complex branch scenarios independently implement the
SELECT formula on the declared address domain and check every small measurement
history, inactive controls, restored inputs and clean work. Padded addresses
are compared against source streams, not an invented identity extension. The
non-Pauli callback also matches the original controlled-H/X stream and both
unnormalized branches of an arbitrary complex superposition.

The controlled-H example demonstrates the difference between equal gate flow
and synthesis estimates: both source and native emitter retain one atomic CH
and four explicit T/T-dagger gates. The source model treats CH as two rotations,
giving 26 T equivalents at its default rate. Those 26 are not an emitted T
count. For the four-entry Pauli example the source estimate and actual T count
both equal 12; there are three measured erasures, two parent clean wires and
six total quantum wires. No data or gate synthesis optimization is applied.

A source-only reproducer records that direct ApplyLthBloq(control_val=0)
emits exactly the same stream as control_val=1 and preserves an input with
control zero. The positive parent API does not advertise that source bug as
negative-control support. An empty inner address axis asserts in the source;
our validation refuses it explicitly. Adjoint dispatch, symbolic extents and
specialized control wrappers remain separate G2 items.

Run `python -m examples.algorithms.indexed_component_dispatch` for the
`circuit.append(op)` example. Seventy-two new tests and the affected original
register, SELECT, block-encoding, walk and control consumers pass together:
278 tests, no skips. The source-reading helper recognizes atomic CZ/CY/CH
wrapper leaves and omits exact identity leaves; no reference source is changed.
Strict mypy covers the emitter and example; Black, Ruff, compiler scope and
architecture checks pass. This is a forward executable slice, not whole-file
acceptance for either upstream class or completion of G2.
