# Declarative and compound fixture audit

Q-0486 and Q-0504 through Q-0507 match the pinned Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. All five unchanged
original tests passed. The native implementations reuse existing
symbolic graph, register Program, tensor and workspace owners.

The declared root has no ports. Stripping atom tags produces the
same seven edges and three leaves, with the original symbolic
multiplicity. The resource queries agree at `100*_n0 + 600` T.
This is a synthetic declaration. The original classical call raises
`NotImplementedError`; the native classical engine has no entry for
this resource root. No physical stream follows from these declarations.

The serial fixture retains all three different atom tags on one
wire. The parallel fixture keeps Split and Join around the same
three actions on independent axes. Both match the original dense
tensor, physical gate operands and the full complex output through
`circuit.append(op)`. The independent fixture has three clean
Allocate/X/Free occurrences and a single workspace slot. The
source and native tensor contractions both project to scalar zero.
The emitted physical stream performs three X operations on that
slot, so its final state is one. Physical workspace binding tracks
ownership and does not project at Free. This keeps Q-0506 open for
the independent fixture's clean-release behavior. Q-0507 covers
the original graph validity tests and separates that boundary.

Each compound fixture emits three X and no T or measurements.
The source assigns a synthetic 300 T and logical peaks of one,
three and one for serial, parallel and independent respectively.
The physical counts and analytical declarations remain separate.

The package entrance Q-0486 is mapped at its seven original
exports. The atom and two-bit action use the native atom fixture.
The fixed-point cast example uses its typed-call fixture with the
previously audited opaque storage boundary. The shaped register
example uses its existing fixture. The remaining declaration,
serial and parallel endpoints use this batch's native fixtures.
The package row verifies those endpoints and their port names.

The concentrated native suite passed 126 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.
