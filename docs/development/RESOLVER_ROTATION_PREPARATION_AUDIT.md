# Rotation state preparation constructor dependency disposition

Reference: complete `state_preparation/state_preparation_via_rotation.py`, its
RotationTree and PRGAViaPhaseGradient dependencies, and unchanged original tests
at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Both Git blobs
match the manifest. Fixtures require actual Qualtran 0.7.0 and Kickmix 0.1 Simulator.

## Ownership and interface

Compiler phase owns compile-time coefficient-to-angle word construction, moved
from the existing Algorithms rotation_tables implementation without duplicating
the algorithm. The Algorithms entry keeps its signature and delegates. This is
constant circuit-data construction; no quantum numerical execution or backend
selection moves into Compiler.

Compiler preparation owns the literal source prefix/control Split/Join graph,
Rx signs and amplitude/phase stage order. Compiler phase composes PRGA using
Allocate, QROM, AddIntoPhaseGrad, QROM and Free. Existing QROM, phase addition and
exact integer arithmetic remain their respective owners. Logical prepare_control,
target_state and phase_gradient are MSB-first THRU ports. Gradient is supplied
by the caller. Angle, carry, QROM scratch and phase flag are private clean work.

Source uncompute selects quantized tables and stage ordering. Exact Adjoint is
independent: the shared graph reverser uses explicit Rx, X and PRGA inverse pairs,
including exact inverse arithmetic. Measurements are never reversed. Constructor
recipes retain those choices within one compilation; unknown adjoints refuse or
delegate to an explicit existing owner.

## Symbolic-width decoder repair

The original phase-gradient-symbolic example serializes and roundtrips upstream
at depth zero. Its constructor stores the symbolic name structurally, while its
redundant register width prints `b_\text{grad}` as text. The native AST decoder
previously rejected that width. It now recognizes the bounded TeX-text atom and
reconstructs the identical existing Symbol. Arithmetic and calls still use the
existing AST whitelist and size budgets. No eval, general TeX parser, public
schema or Core symbol contract is introduced. Attribute access, calls, indexing,
malformed extensions and oversized powers remain rejected. Placeholder names
cannot collide with existing symbols in the input.

The actual symbolic example now decodes, re-encodes and resolves to a bodyless
logical declaration with the original symbol name. Physical execution still
requires concrete width. The fully symbolic HasLength example and depth-one
Shaped payloads fail in the unchanged original codec; those are separate source
boundaries, not substitutes for the repaired native decoder defect.

## Source and execution evidence

The complete concrete original factory and its four PRGA children resolve in
both directions at depths zero and one. Ordered recursive calls, ports and
operands match the source. Comparison stops at the explicitly matched original
atomic AddIntoPhaseGrad boundary. Existing native arithmetic beneath that leaf
is independently executed; a missing source primitive decomposition is not
invented or labeled gate-for-gate evidence.

Small complete complex operators cover zero/one/two controls, both source
uncompute settings and exact Adjoints, with all-history erasure certificates
and full basis-column private cleanup. Source-free compilation covers both the
concrete and repaired symbolic constructors. The eight-bit table checks exact
source calls and phase kickback in both directions with a coherent full address
domain, entangled spectator and caller-owned negative-exponent gradient. All raw
histories are certified; sparse complex outputs verify unchanged gradient and
zero private work. Existing flat source gate/feedback, complete factory/state,
append and JSON evidence is reused in concentrated regression.

## Remaining boundaries

The original singleton omits target_state and has no executable target body;
its logical declaration is retained. Unprovided symbolic coefficient containers,
malformed coefficient admission and arbitrary symbolic controls remain outside
this concrete constructor scope. Source atomic phase-add resource estimates,
source skips/xfails and previously recorded wider QROM allocation-order issues
remain explicit. Generic constructor counts and parent acceptance do not rise
for unverified metadata. No protected public API is changed.

## Acceptance receipt

Final concentrated regression passes 132 cases without skips, including the
eight-bit forward/inverse channel and existing complete original factory/state
scenarios. Local integration passes 145 cases. Strict typing covers eight files;
Black, Ruff, Compiler and Algorithms scope, language, architecture, IR and API
checks pass. Dependency output remains exactly the existing 720 findings.
The complete unchanged original test file has 19 passes, six original skips and
three original xfails; both original notebooks separately pass. No source tests
or reference files were modified.

The original concrete state emits 56 T-family, 122 CX, six Rx, 14 measurements
and 11 peak wires in either direction. Its separate source model retains ten
And, 44 Clifford, ten measurements and 11 logical peak wires. The eight-bit
PRGA emits 52 T-family, 153 CX, 13 measurements and 26 peak wires in either
direction; the source model retains six Toffoli, six And, 42 Clifford, six
measurements and 21 logical peak wires. The native integer adder introduces
carry work and measured cleanup below the original atomic phase-add boundary.
Those physical counts are not substituted for the original analytical model.
Private live leases at return are zero in all four resource rows.

Measured primitive inventories, physical leases and source logical model costs
are separate. The exact PRGA arithmetic inverse is not assumed to have the same
primitive stream as a source leaf that provides no decomposition. Receipts use
`g2-rotation-constructor-*-20261124` under the external reference ledger. The
initial test traversal that repeatedly rebuilt templates was stopped and
replaced with template reuse plus the same literal definition comparison;
no source equality or operand assertion was waived.
