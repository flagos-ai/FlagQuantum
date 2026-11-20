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
ownership and does not project at Free. The complete-source follow-up below classifies this as the source fixture's
violated clean-Free precondition. It does not claim a valid cleaned physical
operation for that fixture. Q-0507 covers
the original graph validity tests and separates that boundary.

Each compound fixture emits three X and no T or measurements.
The source assigns a synthetic 300 T and logical peaks of one,
three and one for serial, parallel and independent respectively.
The physical counts and analytical declarations remain separate.

The package entrance Q-0486 is mapped at its seven original
exports. The atom and two-bit action use the native atom fixture.
The fixed-point cast example now uses its original canonical unsigned
fixed-point descriptor under approved Proposal 115, closed in Q-0489. The shaped register
example uses its existing fixture. The remaining declaration,
serial and parallel endpoints use this batch's native fixtures.
The package row verifies those endpoints and their port names.

The concentrated native suite passed 126 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.


## Complete with_decomposition.py closure

Q-0506 is accepted as a complete source-file mapping. The file declares exactly
three classes, their three signatures and their three builder methods. Each maps
to the existing native serial, parallel or independent Program fixture. Actual
original libraries round-trip bytes and upstream object identity at depths zero,
one and two. Explicit factories resolve the original canonical signatures into
the existing native Programs without importing foreign objects into production.
The same complete operation/operand, lifetime, tensor, coherent append and
independent resource checks are reused for every resolved root.

The serial root preserves its three distinct atom tags and their order. The
parallel root retains Split, three independent atoms and Join. The independent
root retains three Allocate/Atom/Free occurrences and the scheduler's one-slot
reuse. All nine source-library resolutions preserve the original model's 300 T
and logical peaks one, three and one respectively. Measured streams each contain
three X and zero T-family operations or measurements. Model declarations are
synthetic test metadata and are not relabeled as physical synthesis counts.

The former open clean-release item compared two different protocols. The
original independent source tensor contracts to zero, as does the native tensor.
The original Free tensor assumes zero at release, while the source fixture has
just applied X to a newly allocated zero. It consequently violates that
precondition on every independent line. The source Free physical callback only
calls the qubit manager, which is already used by the independent source gate
and lifetime oracle. The native binder preserves that same physical gate stream
and lifecycle. It does not project or reset a released wire under Proposal 103.
The emitted physical state remains one and is explicitly asserted separately.
There is no valid clean-work execution claim for the independent fixture.
Both numerical Free callbacks reject the one value and consume zero. This
preserves the source's invalid-cleanup scenario while closing its representation
mapping. No upstream code or clean-work semantics is altered to manufacture
successful cleanup.

All three original root classical calls stop at the test atom's unimplemented
classical action. Native selected-call evaluation also stops at the atomic leaf,
after the same Allocate or Split prefix where present. The test uses existing
native allocation and integer/split numerical owners for that prefix. It does
not assign a fabricated classical X action to the opaque source atom.

Validation `g2-composition-foreign-*` records 199 concentrated, 17 local
integration and 22 Cirq 1.6.1 passes. All three unchanged original source tests
pass, with original collection warnings retained. Two changed files pass strict
mypy, Black, Ruff, team scope, architecture, IR and public API checks. The
complete source mapping reuses previous native tensor, actual primitive,
measurement-free and work-lifetime evidence. Source-invalid cleanup remains
explicitly recorded above after acceptance of Q-0506.
