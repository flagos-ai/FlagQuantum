# Hubbard SELECT external control constructor closure

The frozen select_hubbard.py at Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389 retains its
[complete stored graph audit](HUBBARD_SELECT_STORED_GRAPH_AUDIT.md).
The constructor intake inspects all five single/grouped/shaped control forms
on both uncontrolled and already-positive-controlled SelectHubbard objects.
All ten forward objects serialize and decompose at the root. Eight direct
adjoint encodings fail on original SpecializeOnCtrlBit enum metadata; the two
single-control-on-controlled forms serialize generic Adjoint wrappers instead.
No frozen source is changed to repair that distinction.

Compiler's existing Hubbard selection factory owns source geometry, direction
and positive-control provenance. Its extension supplies the existing
controlled_via_and_factories positive owner and the existing
multi_controlled_component_pair helper. It does not add another predicate,
allocator, numerical kernel, backend selector or process-wide registry.

Original zero control_val remains refused. Supported outer negative control
uses X / positive SELECT / X. Grouped controls retain source CtrlSpecAnd,
positive SELECT and exact predicate erasure. When SELECT already has its
positive scalar control, additional controls use the original And/SELECT/erase
helper and retain the original control word in the outputs. Packed controls
are row-major, their bits MSB-first, and new controls precede existing controls.
All logical controls, selection and target storage remain caller-owned; private
predicates and child work are cleaned. Exact inverse calls use explicit child
inverse pairs, never reversed measurement instructions.

Unknown or unrelated native children retain refusal. Majorana/spin-up generic
controls and source serialization/decomposition failures remain separate from
these SELECT specialized-control recipes. Existing circuit.append(op), typed
binding and numerical execution owners remain authoritative.

The completed constructor group additionally covers packed two-bit unsigned
controls, singleton bit-array controls and nested new-before-old clauses.
Generic Adjoint of the forward source object remains distinct from the source
object's direct adjoint method: the former can encode the complete native exact
inverse while the latter often fails on the original enum. Deeper source wire
intake separately reproduces an original numpy.uint8 scalar encoding failure;
neither failure is repaired in the frozen source. Successful deeper libraries
compare complete reachable definitions with the existing literal matcher,
including all fields except definition names. They are not compared only by
unitary equivalence. Unrelated same-signature child ownership is refused and
symbolic SELECT outer controls retain logical declarations and physical refusal.

Complete recursive source call/operand/feedback checks cover executable source
control states and both directions. The already-controlled nested root fails
in the frozen specialized control adder: np.concatenate receives an array as
its axis argument and raises TypeError. Its reachable extended specialized
child is independently compared to the executable source child, including all
recursive calls. Native outer composition is additionally checked against
activation semantics; the failing source root is not claimed executable. Complex sparse states include every outer
control pattern, both original control values, logical data and a correlated
spectator. Each added predicate's measurement outcomes are exhausted while
retaining the child channel's certified histories; full-zero/full-one/alternating
raw histories and forward/inverse restoration are checked as applicable. The
whole-stream erasure certificate proves the unchanged channel for every measured
history; no claim is made to a brute-force enumeration of every child history.
All populated amplitudes, absolute relative phases and private clean outputs
are compared. Ordinary circuit.append(op) retains the exact emitted payload.

The twenty-eight directional resource rows independently compose the existing
controlled/extended-controlled component model around the positive SELECT
model. All twenty-eight source QEC category rows agree. Twenty-six source logical
peaks agree; the two nested already-controlled source peak queries fail with
the same original concatenate defect and are retained as failed boundaries. Source asdict omits
zero categories, so equality uses the same nonzero projection of the native
model while retaining both raw dictionaries. Named physical gates, measured
feedback, private simultaneous live peak and full supplied pool remain separate.
The borrowed spectator is one extra physical wire, excluded from operation
logical/private peaks and included in the full actual live peak.
Forty-eight complete old constructor Programs, including singleton declarations
and exact base inverses, equal published a4b977598.

Final batch validation passes 148 concentrated cases in each Cirq 1.7 and 1.6
profile and 145 local integration cases. Strict typing covers the factory and
two test modules; format, lint, owner scopes, language, links, architecture,
IR and protected API checks pass. Repository dependency policy retains 733
findings, none on this batch. This constructor closure does not close the
whole source-file parent while generic child control and capture scope remain.
