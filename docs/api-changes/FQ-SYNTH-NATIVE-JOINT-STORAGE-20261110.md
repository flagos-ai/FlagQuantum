# Joint native register preparation and matrix targets

The authorized native corpus migration next requires the unchanged Lanchester
HHL input: one entangled real vector across a one-bit group field and a four-bit
time field, followed by matrix QPE over those same fields. Separate preparations
would change its state. Replacing both named outputs by an opaque five-bit output
would weaken readout. Existing examples flatten source fields to named native
ports; retain that pattern and their numeric metadata.

Broaden the existing `unitary`, `inplace_prepare_amplitudes` and
`inplace_prepare_state` target types to a flat static sequence of whole QNum,
QArray or QBit ports and existing checked array views. Sequence element zero is
least significant; each element contributes its existing least-significant-first
locations. All expanded locations must be distinct, initialized and writable.
No new owner, allocation, reset, reinterpretation or register publication occurs.
Single whole-register calls preserve behavior. Direct inplace subviews remain
refused; an explicit sequence makes joint or partial storage selection visible.
Nested sequences, empty/overlapping targets and uninitialized/Const storage fail
closed. Existing mutable-helper borrowing and pure inverse-region checks remain.

Reuse indexed qreg.inplace_prepare_amplitudes and qreg.unitary, their existing
verifiers, inverse rules and lowering. Real vector lengths remain 2..256 and
matrix dimensions remain 2..32. Existing memory/workspace/operation/depth budgets
apply. No new IR opcode, numerical kernel, record class or root export is needed.
Old admission contracts remain historical; a successor candidate specifies only
these three broadened signatures. The API checker must verify both records.

Acceptance requires independent complete complex Circuit vectors at original
Lanchester input and two nonzero variations, unchanged matrices/rotation tables,
exact numeric labels for both fields, source bit order, absolute phase, clean
matrix decomposition work and real budget refusals. Smaller entangled, signed
fractional, nonzero-column, spectator, slice and inverse cases exercise the
reusable operation rather than one hardcoded example. Fresh216source replay,
immutable evidence and catalog/ledger updates precede native coverage credit.
Only local implementation is authorized; no release, hardware, performance or
full Notebook workflow claim follows from this admission.
