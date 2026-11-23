# Conditional addition constructor dependency audit

## Complete source scope

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Complete
`arithmetic/controlled_add_or_subtract.py`, its original tests, Add, BitwiseNot,
Controlled and resolver dependencies are covered by six frozen Git blobs in
`ftqc-reference-env/g2-conditional-constructor-intake-20261123.json`.
Prior complete source acceptance and exact-adjoint evidence are reused from
CONTROLLED_ADD_OR_SUBTRACT_FILE_AUDIT.md and the native conditional component suites.

Compiler `formats/legacy_arithmetic_factories.py` binds the complete
ControlledAddOrSubtract constructor through the existing
`registers/arithmetic/conditional_addition_component.py`. The source signature
is ctrl/a/b, with default b retaining a's dtype. Source truth-valued polarity selects
whether control one adds or subtracts. All words are MSB first and remain
caller-owned outputs. Concrete unsigned, signed, Montgomery and mixed words
retain their metadata. Symbolic signatures remain logical declarations without
physical execution.

## Scoped controlled complement dependency

The source root contains two Controlled(BitwiseNot) calls around Add, plus two
control X calls when add_when_ctrl_is_on is true. Word constructor adaptation
now retains exact resolved complement definitions within one compilation's
existing catalogue. Its scoped Controlled factory accepts only those definitions
and one positive scalar QBit clause. Unknown children, non-scalar, negative,
non-QBit and multiple-clause specifications refuse. Generic Controlled remains
open. The source constructor identifiers remain interchange metadata.

Concrete complements delegate the existing controlled_bitwise_word_program.
Symbolic complements preserve the ctrl/x declaration. No helper body is inferred
from a function name and no external object enters Compiler. Retained recipes
are bounded by the catalogue lifetime, matching existing conjunction-adjoint
constructor context. Existing Add, OnEach, X, bookkeeping, numerical codecs,
component control and execution owners keep their responsibilities.

## Source comparison and execution

Named source libraries and directly decomposed libraries are compared with the
unchanged original primitive dependency order, operands and measured feedback.
Both polarities, equal/unequal words and reversed physical word placement are
covered. Every small raw history matches the complete unnormalized source
complex tensor with an entangled spectator and zero private work. Complex64 and
complex128 trajectories run through ordinary circuit.append(op). Existing
opposite-polarity inverse-history and specialized outer-control evidence is
reused. All three original concrete/symbolic examples resolve at depths 0/1/2.

| Quantity | Actual primitive inventory | Independent source model |
|---|---|---|
| T and T-dagger | 4(n-1) combined | QEC total T count 4(n-1) |
| Measurement and reset | n-1 of each | QEC measurements n-1 |
| Controlled complements | 2n CX in total | Model categories retained independently |
| Control flips | 2 X for positive add polarity, zero for opposite | Independently queried original model |
| Clean private work | n-1 Add carry wires | QubitCount agrees for tested concrete widths |

## Remaining boundaries

Generic serialized Controlled/Adjoint/Power and arbitrary controlled word
constructors remain open. The scoped positive complement dependency establishes
this source file's constructor closure. The source does not validate its polarity field. The format adapter retains
its original Python truth-value conversion, including serialized nonbinary
numbers and strings. Ambiguous array truth values retain the original failure.

Original narrow signed classical/gate disagreements and the one-bit Clifford
model undercount remain in the complete file audit. Negative outer control can
hit the source Controlled(And-erasure) unitary recursion failure. Symbolic
physical binding, pending Proposals 116-119, whole-repository checks and
GPU/performance coverage remain open. Source bytes are unchanged.

This batch brings complete built-in coverage to 42/303 identifiers, with 261
remaining. The partial Controlled mapping does not count as a complete identifier.
Q-0876 remains open. Batch receipts use
`ftqc-reference-env/g2-conditional-constructor-*`.

## Batch verification

The final coverage selection passes 320 cases with 100 percent statements and
branches across both affected constructor owners. Affected local integration
passes 145 cases. All four changed Python files pass strict typing, Black and
Ruff. Compiler scope, language, architecture, multilevel IR and protected API
checks pass. Dependency findings are identical to the 720 recorded baseline
findings, with the checker retaining exit one.

The unchanged original test file passes twelve cases and retains one original
count-mode skip. Six frozen blobs and native Kickmix 0.1 Simulator are verified.
Prior complete file, notebook, native inverse-history and SDK evidence is reused.
Original protobuf deprecations remain visible. These checks do not establish
whole-repository or GPU coverage.

The final concentrated direct-dependency regression passes 1826 cases without
skips. The focused constructor selection passes 71 cases. Full primitive
histograms and independent original models remain separately asserted.
