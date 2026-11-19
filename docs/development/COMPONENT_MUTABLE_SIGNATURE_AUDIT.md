# Mutable component signatures

The remaining construction path in Q-0156 is the source builder's ad hoc
signature mode. Compiler owns this mutable construction state. Core retains
the immutable Register, Signature, FunctionRecord and ProgramRecord contracts.
Existing callers keep strict construction by default. The internal constructor
adds a keyword-only `allow_register_addition=False`. Explicit opt-in permits
`add_register` and infers missing RIGHT registers during finalization.

Input arrays use row-major element order, with one SSA value per element even
for zero-width words. New input values can follow previously emitted calls;
the final block argument tuple still contains only input values, in signature
order. Supplied Register metadata is retained. A string accepts a concrete
integer width or a RegisterType. Width one selects bit, other widths opaque.
RIGHT-only additions return None and create no input handles. Adding registers
does not allocate physical workspace.

Output inference inspects the scalar binding or first flattened array element,
preserving source dtype, shape, output order and failure timing. Empty inferred
arrays fail at element lookup as upstream does. Explicitly declared empty
arrays remain supported. Duplicate names are recorded and rejected by Signature
at finalization, matching source timing. Existing consumption and finalization
checks remain authoritative. Symbolic typed Program admission is still tracked
under Q-0149 and is outside this finite construction implementation.

Negative widths differ explicitly. The source accepts a negative-width register
as metadata; the existing native finite Program boundary refuses it before
creating a handle. Non-binding inferred arrays raise a native ValueError before
consumption, rather than a source attribute lookup error. These are recorded
admission differences, not equivalent source outcomes.

The source/native acceptance scenarios cover scalar, vector, matrix, empty
arrays and zero-width words, renamed inferred returns, exact post-call input
ordering, duplicate-name validation, inference failure prefixes and all three
typed/width/explicit argument forms. Two CNOT primitives preserve complete
complex amplitudes and an untouched spectator through `circuit.append(op)`.
Their source resource model also counts two Clifford operations. Zero-primitive
allocation transfers two clean wires through an inferred output; a separate
LEFT-only input-release case frees the caller-owned three-wire register.

The concentrated builder, graph, rebinding, inlining and complete notebook
regression passes 158 cases without skips. Strict typing checks the three
changed Python files. Black, Ruff, Compiler ownership and architecture checks
pass. Output inference has 17 statements with 100% measured coverage. The full
dependency scan retains 706 pre-existing findings, with zero changed-path
findings. Q-0156 remains open until its remaining source APIs and symbolic
boundaries are accounted for.
