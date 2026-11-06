# Concrete QROM lookup

`source_qrom_ir` binds caller-owned controls, big-endian address registers
and row-major datasets to the locked Qualtran QROM and unary iteration
decomposition. Each dataset can contain several same-width words per
address. It XORs those words into the corresponding target and preserves
the controls and address. The source QROM is its own logical adjoint.
Applying the emitter twice with clean scratch restores the target.

The bounded tree trims unused halves, collapses source-uniform subtrees
and reuses one And work wire per active depth. Uniformity is tested over
the original flattened slice of each dataset, including its target words.
The zero-selection path loads a single row directly. Multiple positive
controls use the original And ladder. Both the ladder and tree are erased
with measurement, conditional CZ and reset. Callers bind the clean work
and one reusable classical slot per reserved work wire.

The executable source defines a channel on valid indices. A pruned
address can alias a populated row, so callers requiring zero outside the
domain must explicitly pad their table. Nonuniform data that mixes a
singleton axis with a live axis fails in the locked source after the
singleton is removed from its signature. Zero-width targets also fail
when the source attempts to decompose XorK. The emitter rejects both
layouts. An uncontrolled scalar lookup uses no work. The source static
qubit estimate subtracts one qubit for that case and reserves the full
address pool even when a uniform table short-circuits. The emitter reports
the actual concrete allocation separately.

A one-control lookup with three two-bit words uses 8 T-family gates,
18 CX, 2 measurements and 2 clean work wires across 7 qubits. The
independent source T and measurement estimates agree. Tests compare
original operands, per-wire order and feedback, certify the complete
clean-work channel on all valid address and target inputs, and directly
project every measurement history of the tested small circuits. These
projections include complex amplitudes entangled with a spectator and
retain the unnormalized branch probability. Source parity covers nested
and truncated axes, constant subtrees, shaped target groups and up to
three controls.

The test reader reshapes source ports according to their signatures. It
uses the original MultiAnd generator on the explicit junk and target
wires to avoid the Cirq export's temporary replacement output allocation.
Its inverse comes from Cirq's inverse of the original logical And ladder.

The concentrated dependency regression passed 295 tests. The final
19-case module run reaches 100% statement coverage. A separate 118-case
regression checks the existing source reader consumers. The source QROM and
unary iteration blobs match the pinned inventory. The paired lookup
regressions exercise the existing G1 consumer path.

## Generic outer control

`control_source_qrom_ir` accepts a structured outer predicate, separate
from the lookup's existing positive controls. The caller supplies its
clean predicate wires and measurement slots. `source_qrom_ir` also
accepts a single positive `outer_control` for a containing component
that already owns the predicate. SelectSwapQROM uses this shared path.

The generic controlled lookup preserves the emitted X and CX order by
mapping them to CX and CCX. A load that emits an And computation or
erasure is rejected because the pinned generic controlled versions of
those components cannot decompose their non-through registers. This
check follows the concrete load, so a reserved but unused QROM work
wire does not prevent a pruned lookup from being controlled.

For the two datasets `(1, 1, 2, 2)` and `(0, 0, 1, 1)` with outer
pattern `(1, 0, 1)`, the controlled load uses three CCX instructions.
The outer predicate adds eight T/T† and two measurements and clears
both work wires. The caller reserves 11 total wires including the
unused lookup work wire, while the source peak estimate is 10.
