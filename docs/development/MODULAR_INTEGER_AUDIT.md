# Modular scalar value source audit

## Ownership and interface

The frozen source is Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0,
`util/mod_int.cc`, `mod_int.h` and `mod_int.test.cc` (K-0312 through K-0314).
Core owns the private immutable `ModularInteger(value, modulus)` scalar.
Both fields use the accepted `FixedWidthInteger` owner. Shared immutable modulus
identity survives successors and copies. Direct construction represents source
raw state without implicitly normalizing it. `from_signed`, `from_text` and
`from_word(..., moved=...)` retain the distinct source constructor routes.
Raw integer word order is little-endian, bit zero is the least significant bit.
No quantum storage, measurement history or work qubit belongs to this type.
Simulation owns random draw consumption using the existing fixed-width sampler.
Compiler continues consuming existing scalar and register interfaces.

Updates return successors and leave caller snapshots usable. Source copy/move
assignment maps to immutable rebinding. A moved-from C++ object has invalid
storage and has no native logical value. Const and separate-object rvalue
division use the same source algorithm on owned snapshots. Explicit C++
self-rvalue division maps to `divide(self, moved=True)`. Its starting cofactor
has zero-width moved-from storage. Source results can retain that invalid width.
Nonunit division throws after source mutations. Native failure preserves caller
snapshots. Source post-failure state must be retained in the comparison fixture.

Equality compares the value alone. Cross-modulus arithmetic uses the target's
modulus. Signed construction clips before negative handling, text construction
reduces before copying. Negative text fails original parsing. Add performs a
single correction and retains the accepted scalar carry behavior. Division
follows the binary cofactor loop directly. Signed shifts repeat source modular
doubling/halving, including reversed direction for negative amounts.

## Intake evidence

All six complete source/direct dependency blobs are Git-blob verified in
`ftqc-reference-env/g2-modular-value-intake-20261122.json`. All 20 unchanged
original tests pass. The checked-in generator independently verifies seven
source blobs, compiles the unchanged C++ sources and records 12,442 observations.
There are 10,516 successful values, 1,781 failures and 145 equality comparisons.
Widths 0/1/2/3/4/7/63/64/65/127/128/129/256 include hanging bits, unequal
constructor widths, cross-modulus operations, both shift directions, all integer
shift overloads, scalar boundaries and repeated seeded sampling. Output retains
raw storage, declared width, invariant status, truth value, decimal rendering
and the next RNG word. Failure records retain the original mutated target.

## Complete definition mapping

| Frozen definition group | Native owner |
| --- | --- |
| Deleted default, zero, signed, text, const-word and rvalue-word construction | Core ModularInteger explicit fields and constructor factories. Direct fields represent raw source state. |
| Copy/move construction and assignment, shared modulus pointer | Immutable snapshots and rebinding. Successors retain modulus object identity. Source moved-from storage is represented only where the division algorithm observes it. |
| Invariant checks, bool and non_zero | verify_invariants and bool. Allocated word count follows declared width and cannot independently disagree. |
| Value-only equality | __eq__ and consistent native hashing. Modulus equality is deliberately excluded. |
| ModInt/scalar addition, subtraction, increment and negate | add, subtract, increment and negate reuse exact source word carry/borrow/clipping order. |
| ModInt/scalar multiplication and squared | multiply and squared accumulate into the source temporary width, perform source remainder and retain low allocated words. |
| Const/rvalue/scalar division and inversion | divide with moved ownership and inverse retain the original binary cofactor loop, even-modulus refusal and nonunit failure. |
| Double/halve and all four signed/unsigned shift overloads | double, halve, shift_left and shift_right reuse accepted Core modular words. A Python amount preserves the numeric value of its selected source overload. |
| random/randomize | Simulation sample_modular_integer composes the accepted scalar rejection sampler. Repeated calls retain caller-owned RNG state. |
| ostream and str | __str__ composes accepted scalar decimal conversion and source modular formatting. |
| All 20 original tests | Unchanged original execution plus the complete source corpus and native ownership tests. |

## Source defects and ownership boundaries

`value *= INT64_MIN` negates the signed minimum in mod_int.cc line 166.
UBSan reproduces signed overflow. Native multiplication refuses that input.
The [2026-11-28 review](G2_FUNCTIONAL_MIGRATION_DISPOSITIONS.md) accepts K-0312's
complete defined-domain migration with this original defect retained. Receipt
`ftqc-reference-env/g2-modular-value-sanitizers-20261122.json` records the actual
sanitized source failure and the successful self-move division probe separately.
The original reference is unchanged.

For value 1 modulo 3, division by zero throws after changing the source target
to zero. The native call throws with the original immutable snapshot intact.
The fixture records the source post-failure value. Self-move division can
return a zero-width value that fails invariants, while ordinary self-division
uses its original const-copy route. Both successful source results are compared.
An unrepresentable independent word-count corruption has no native state.
Huge shift counts retain linear repeated source operations and have no execution
time guarantee. Bounded shifts are covered. No performance acceptance is claimed.

## Resource accounting and verification

These are classical scalar values. Actual emitted quantum primitives,
measurements and work qubits are zero. The source defines no quantum resource
model for this class. Existing physical register consumers retain their accepted
complex-amplitude, measurement-branch and work-zero evidence. No additional
quantum decomposition is inferred from scalar equality.

All 120 concentrated tests and 92 local register integration tests pass without
skips. The new scalar module has 100% measured statement and branch coverage
(131 statements and 50 branches). Strict typing passes for all ten changed and
direct Core dependency Python files. Black, Ruff, both owning team scopes,
architecture, multi-level IR and public API checks pass. The full 12,442-record
source corpus regenerates byte-for-byte. Dependency policy retains 720 existing
findings with zero touched-path findings. Full default-suite and ambient
dependency acceptance retain their previous separately recorded boundaries.
An initial test insertion placed an assertion in the next test's scope. That
test-only error was corrected before the passing final concentrated run.
