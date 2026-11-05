# UP-06 compact Pauli rotation contract

This is an unreleased private construction-profile extension. Public exports,
`Program`, execution results and Stable Core serialized contracts do not change.
Ecosystem translates Qmod data immediately into existing owned HybridProgram
records. Compiler alone maps Pauli rotations to physical instructions.

## Semantics

`qreg.pauli_rotation(effect, target[, control]) -> effect` has exactly `support`
and `angle` attributes, plus `control_index` when a control operand is present.
Support is an immutable ordered tuple of `(numeric_bit_index, axis)` pairs.
Axes are X/Y/Z; indices are distinct non-Boolean integers within the target.
Bit zero denotes its least-significant numeric bit. Empty support denotes I.
The real finite float angle `a` denotes `exp(-i*a*P)`, including the absolute
phase `exp(-i*a)` for empty support. Twice the angle must also remain finite.
An optional scalar bit in an initialized disjoint register controls the complete
unitary, including its identity/global phase. The control index is validated.

The target must be initialized and writable. Control is read-only. Effect
ordering, borrowed-storage checks, dependency invalidation, qubit allocation and
return ownership retain their existing authority. No runtime state inspection,
random sampling, new scratch allocation or backend choice belongs to lowering.
Adjoint changes `a` to `-a`; source-order Trotter products are never reordered.

## Bounds and compatibility

Qmod source acquisition is capped at 512 KiB and 65536 tokens. Expression,
scope, definition and owned-node budgets remain independent and unchanged.
One Pauli factor occupies one owned effect; the 4096 owned-node budget is not
raised. qDRIFT accepts 1..2048 literal samples, preserving all 2000 samples in
the original example. The existing local deterministic profile uses seed zero,
source-order terms, absolute coefficient weights and the same sampled sequence.
Sampling remains in the adapter, not the Compiler. No operator terms are dropped.

The physical mapping migrates the established H/RX, CX parity and RZ template
into one authoritative Compiler implementation. The adapter's previous expanded
template is removed. Identity mapping preserves global phase. Existing admitted
circuits must keep their semantics and logical costs, or any measured cost
change must be independently checked and recorded without editing old baselines.
The final 100000-instruction bound, user gate/depth/qubit/workspace constraints
and execution-memory limit remain enforced. Expanding one bounded support is
linear in target width and at most 1023 instructions before its budget check.

## Acceptance

Use the two byte-identical originals, each with 551 terms and 12 target bits.
Keep the original first-order Trotter schedule and all 2000 qDRIFT samples.
Record source and license hashes and the full sampled-index-sequence digest.
Compare complete complex128 state vectors against an independent Circuit with
the same fixed sampling sequence; do not align away global phase. Check I/X/Y/Z,
numeric bit order, noncommuting order, negative angles, coherent input and
controlled identity phase on bounded synthetic cases. Validate owned records
independently of the adapter, rejecting malformed support, nonfinite angles,
uninitialized or Const targets, aliases and invalid control indices.

Test source/token/sample/owned-node and final resource-budget refusals. Recheck
all 216 original paths and preserve old resource evidence. Retain executable
Qmod, native qfunc and Circuit examples with their actual supported scope.
No integration PR or full CI is opened until the agreed corpus gate is complete.
