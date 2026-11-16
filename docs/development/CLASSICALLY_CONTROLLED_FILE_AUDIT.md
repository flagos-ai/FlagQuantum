# Classically controlled component audit

The declaration and its original test are pinned to Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native Compiler
register owner prepends classical control metadata, preserves declaration
order, and resolves name collisions with the same numbering. When a
child already owns `ctrl` and `ctrl2`, the two added control names
are `ctrl3` and `ctrl4`. Shapes and classical data types are retained.
Quantum control metadata is rejected by this entry.

The native typed feedback function applies the unchanged bit-gate
lowerer for a coherent child. Each CBit input is read through an existing
owned snapshot operation and its successor stays in the same physical
classical slot. Conditions are attached to primitive gates in their
original order. The attachment uses ComponentProgramBuilder and
returns control successors in control declaration order, followed by
the child's ordered outputs. The existing one-bit CZ consumer uses
the same constructor. Register shape, execution admission and
classical-slot validation remain with the existing owners.

## Test-file acceptance

Q-0548 closes. The pinned source test passes and all four of its
scalar truth-table inputs run through append with the source control
value forwarded. The shaped test builds the original attachment
callback and its native counterpart with two CBit array entries and
one scalar CBit. A three-measurement native program evaluates all
eight histories against the corresponding locked classical action
for every computational child input. Every unnormalized complex
amplitude is compared. The emitted child is one X with an ordered
three-slot guard, with no extra primitive for control reads and no
quantum scratch. Both child wires and each classical output slot
retain their bindings.

The prior source tensor, phase, and measurement-branch work is reused,
as are the measurement-uncomputation and field-erasure consumers.
The concentrated regression passes 529 tests.

## Remaining implementation boundary

Q-0547 remains open for typed Program execution with wider classical
data types. Wider-word metadata and existing flat IR guards remain
available. The current typed feedback read accepts owned CBit
elements. The new adapter takes an unconditional coherent gate body.
The separate flat IR path still composes existing guards. The flat IR rejects conditional
measurement, reset, and classical writes.
