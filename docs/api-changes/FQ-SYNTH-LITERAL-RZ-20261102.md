# Literal RZ compatibility after UP-00 review

Owner: Ecosystem source admission and Compiler register lowering.
Status: unreleased #1003 compatibility slice, governed by the
[UP-00 unified-program contract](FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md).

Follow #1002 at 6df1f2ae6296334d087fdf39918f8b1e46559a19. Run pinned RZ.qmod
unchanged at original size with file/text, source/license and full-state tests.
Coverage is eight of 216 original paths and seven of 208 distinct blobs only
where original-size execution is verified; frontend admission alone is not that
evidence. This task adjusts the PR without merging it.

## Retained compatibility

Compiler reuses the existing qreg.gate angle contract for phase/rz: exact
attributes, finite Python float radians, initialized target and validated index.
Existing mutation/Const/effect/budget rules remain authoritative. Lower to Core
rz/phase with named theta; no new opcode, matrix, executor or numerical kernel.

Ecosystem admits nonnegative integer/contiguous fixed-decimal literals only in
the first PHASE/RZ argument. Reuse token/expression records; no decimal data
arithmetic. A literal has at most 78 characters. Check its exact Decimal value
against 0..2**53 before float64 conversion, including rounded overshoot guards.
Preserve parenthesized integer PHASE. Refuse negative, exponent-form, split,
symbolic, computed, quantum, nonfinite and out-of-range source angles, and
numeric/array/Const targets. Both gate names are reserved intrinsics.

PHASE decimal admission intentionally extends the previous integer-only source
profile. RZ(theta) differs from PHASE(theta) by global phase: compare absolute
complex amplitudes, including coherent and reference-entangled states. Do not
normalize away this difference. Tiny nonzero angles must remain nonzero.

## Withdrawn prototype and native gap

Head 4c923bf111ace9c29bf0d784bbfabfc0fee940d9 also introduced one-UInt(1)
float-return phase capture. UP-00 withdraws that unmerged prototype rather than
freezing a new effect inferred from return type and Services-specific dispatch.
Its 118-test and resource evidence remains historical, not current admission.
Existing Python int/bool/unannotated pi predicates and pure arithmetic remain.

A unified `qfunc` with explicit `phase(expression, coefficient)` is the design
baseline; it is not implemented by this RZ slice. A Qmod RZ user is not required
to rewrite gates as a float-return Python function. Every delivered example
separates current native high-level functionality, Qmod import and Circuit
reference, with the arbitrary-angle native phase gap stated explicitly.

Acceptance: original pins/license; file/text, objective and preparation variants;
RZ/PHASE absolute phases; helper/role/initialization/malformed IR failures;
exact Decimal bounds; target/budget failures; previous Python behavior and the
withdrawn float profile's early refusal. See the
[delivery](../roadmap/SYNTH_LITERAL_RZ_DELIVERY.md) for measured evidence.
No general CReal, unified native effectful program, all-corpus, release or
hardware admission. Independent trials remain 0/2, release_authorized=false.
