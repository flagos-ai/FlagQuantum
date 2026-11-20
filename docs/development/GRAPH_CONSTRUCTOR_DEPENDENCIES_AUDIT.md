# Live graph constructor dependencies

The reference is unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. This dependency belongs to the
complete graph serialization and controlled-component source files, Q-0866 and
Q-0158, and their ECWindowAddR consumer Q-0458.

## Ownership and correction

Ecosystem converts the optional live graph to owned Program definitions.
Compiler retains component resolution, calls, typed bindings and execution.
No foreign object survives in the returned record.

The previous importer serialized each leaf independently and only registered
actual graph nodes. Two controlled MatrixGate children with different matrices
both acquired child reference ID 1 in their separate declarations, producing a
false ambiguity. A controlled QROAM node instead failed because its constructor
child had no registered native factory.

The importer now uses the same unchanged source discovery functions and depth
as the graph serializer. Its live object IDs match the complete library IDs.
All constructor child definitions are available to the existing resolver.
Resolved child arguments must name the exact native definition corresponding to
the original reference ID. Scalar and array arguments still require canonical
owned equality. The Montgomery raw Split/Join exception retains its earlier
scope. Arbitrary live declaration collisions still reject.

## Acceptance

The baseline reproduces the false collision for two distinct actual controlled
MatrixGate children. Native conversion now preserves both children and relative
phase, both control polarities, nested controls and the available original
adjoints. All basis states and a phase-sensitive coherent state with spectators
are compared against the original full tensor on reversed physical wire bindings.
JSON roundtrip, append and subsequent execution run with source imports forbidden.
The original serializer's Git blob is verified against
`132930b2804536155f5b242dbede88518fcca459` at runtime.

One original negative nested-control graph adjoint fails before conversion with
`BloqError` requiring a Soquet named `ctrl1`. Its actual error is retained as a
positive diagnostic test. Seven other matrix executions compare complete complex
outputs. The source has two opaque controlled matrix calls. Native execution has
two explicit unitary primitives, plus four X primitives in the negative-control
adjoint representation. There are zero workspace events and measurements.
These matrix fixture primitives have no independent source QEC model here.

## Actual windowed point consumer

Controlled ECWindowAddR now resolves and imports the actual QROAMClean constructor
child. The next controlled QROAM node enters the existing tensor-backed matrix
fallback. A 30-second bounded intake reaches native `matrix_error` unitary
validation after constructing that node's dense matrix. The receipt records the
completed child and the unfinished controlled node with a stack trace. This is a
native graph-import performance boundary, not evidence that the source controlled
QROAM has failed or that the complete consumer executes. The earlier unbounded
probe was interrupted. No complete ECC parent is accepted from this dependency.

Receipts in the reference directory include
`g2-graph-constructor-probe-20261121.log`,
`g2-graph-constructor-regression-20261121.log` and
`g2-graph-constructor-ecc-followup-20261121.json`.
