# Clean QROAM constructors and direct dependency disposition

Reference: unmodified Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`data_loading/qroam_clean.py` constructor group and its QROMBase,
SelectSwapQROM, unary iteration, Partition and SwapWithZero dependencies.
The focused fixture checks all four data-loading blobs against their locked
Git object hashes, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator.

## Interface and ownership

qroam_clean_program composes existing typed Partition, QROM and SwapWithZero
Programs. It reuses block packing, XOR traversal and CSwapApprox owners.
Selection/control ports are THRU. Selected targets and retained junk are RIGHT
outputs whose storage transfers to the caller. Dataset and bank order are
row-major; bits are MSB-first. Each dataset allocates junk cells then its first
selected target as in the source. Predicate work is private and returns zero.
The existing scalar lookup/measurement-erase component remains available.

The three explicit constructor factories retain the original concrete data,
widths, shapes, control count and block metadata. Forward examples resolve at
serialized depths zero and one to the same native module. Wrapper recipes stay
local to one compilation. QROAMCleanAdjointWrapper reshapes each target plus
its retained junk into the original block-shaped leaf input. Its actual single
source call and operand grouping are checked. QROAMCleanAdjoint has no original
gate body: native resolution retains its typed bodyless declaration and physical
binding refuses it. Native measurement erasure is separately tested and is not
substituted for that missing source decomposition. Production factories also
compile serialized lookup and wrapper records with source imports blocked.

## Source scratch lifetime fix

The first multi-dimensional comparison exposed a real physical operand gap.
Directional native And calls released a predicate slot before the enclosing
source unary generator released its preallocated scratch word. Thus the last
two native erasures used wires 47/46, while the bound original used 48/47.
Numerical agreement was not used to waive that strict failure.

The shared unary call owner now supports a reserved source traversal word.
And operates on already-owned target slots through the same four-T compute and
exact measured-erasure emitters. Existing Join/Split views return one ordered
loop frame containing selectors, control and scratch before freeing its word.
Those SSA dependencies keep scheduling from releasing erased slots before a
later nested leaf returns. The views add no quantum gates, API vocabulary or
IR types. The reservation is requested by this concrete QROAM composition;
the existing directional logical-call path retains its contract.

Both original examples now pass strict primitive counts, per-wire operand
streams and measurement/feedback slot correspondence. The multi-dimensional
case is no longer normalized by renaming workspace. Sparse complex execution
covers all 25 valid addresses, and the shared symbolic certificate proves all
128 erasure histories have the same normalized action and restored scratch.
Additional controlled, unblocked and multidimensional cases check source
physical placement. Large bank states use sparse execution; sparse floating
residual amplitudes are compared with numerical tolerance, while the primitive
operand assertions remain exact.

## Forward execution and native erasure

Small new-forward/existing-native-erase compositions exhaust all raw histories
for both block settings, including an entangled spectator. Each branch returns
the complete complex input scaled by its projection amplitude. No work leases
remain live. This is explicitly native executable erasure evidence, independent
from the original model-only leaf. Register callers retain the existing
circuit.append(op) execution workflow; no external SDK objects enter Core IR.

## Measured resources and original models

For the original multi-data example, emitted primitives are 14 T, 14 Tdag,
51 H, 53 CX, 25 S, 24 Sdag, 4 X and one each measurement, conditional CZ and
reset. Actual peak allocation including inputs is 16 wires, with 12 caller-owned
output-bank bits. The original forward model separately declares 24 T, 1 And,
146 Clifford and 1 measurement, with peak 16 logical wires.

For the original multi-dimensional example, emitted primitives are 74 T,
74 Tdag, 261 H, 334 CX, 127 S, 120 Sdag, 16 X and seven each measurement,
conditional CZ and reset. Actual peak is 49 wires, with 40 caller-owned bank
bits. The original model separately declares 120 T, 7 And, 781 Clifford and
7 measurements, with peak 49 logical wires. Existing native model formulas
match the original categories and widths; logical And nodes are not reported as
already-emitted T gates. Full resource receipts are in
`g2-qroam-constructor-resources-20261124.json` outside the repository.

## Retained original defects and boundaries

- The multi-dimensional original wrapper reports seven Toffoli, while its
  padded adjoint leaf reports nine. The wrapper uses original 5x5 data extent;
  its leaf uses the padded table. The one-axis example reports one in both.
  Both source estimates remain independently verified and unchanged.
- A whole-table bank on four entries with log block size two fails original
  graph finalization because `selection` is missing. The existing native
  whole-bank capability remains separately checked against its accepted raw
  emitter; it is not presented as an available original source graph.
- The erasure leaf is model-only upstream. Unknown or unowned wrapper context
  refuses. Source symbolic/object-array serialization, QROM admission gaps,
  mixed omitted selectors and shaped-target boundaries remain explicit.
- No complete generic catalogue identifier or parent item is added for this
  concrete disposition. Proposal 116 is approved and implemented. Proposal 119 remains unapproved
  and is not required by this source closure. Broader G2, GPU and performance
  scope remains in the central ledger.

## Acceptance

The concentrated lookup/unary regression passes 193 cases without skips and
related local integration passes 145 cases. The complete unchanged original
QROAM test file, including its slow classical multi-dataset case, passes 25,
with one source-owned qtyping skip and one source-owned counts xfail in the
existing Cirq 1.6 compatibility environment. Source counts enforcement remains
an upstream limitation. Strict mypy checks seven Python files. Black, Ruff,
Compiler scope, language, architecture, IR and public API checks pass.
Dependency analysis matches the 720 existing findings exactly.

Q-0477 is accepted under the user-approved 2026-11-27 faithful-migration
rule. The source failures above remain independent follow-ups; see the
[complete-file disposition](QROAM_CLEAN_FILE_AUDIT.md#faithful-migration-acceptance-on-2026-11-27).
