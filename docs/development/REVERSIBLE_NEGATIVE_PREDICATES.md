# Negative conjunction legalization

G2-KCFB03 extends the existing compiler-owned predicate binding. Core still
validates the original complete program. One canonical conjunction can contain
zero and one values. Disjunctions remain outside this slice.

Before an instruction or recognized effect macro, compute each negative guard
into a distinct compiler-owned classical complement: BIT_STORE0 scratch,
BIT_INVERT scratch, BIT_INVERT scratch if original. The scratch block starts
above every original slot, including unowned holes. Positive guards keep their
original IDs. Use the resulting positive guards with the existing atomic parity,
quantum gate and HMR/R paths. Never invert a caller guard in place.

Scratch lifetimes end after the instruction/macro and its native scopes close,
so subsequent instructions reuse the same block. Declared scratch width is the
maximum number of negative tests in one conjunction, not their sum. Recompute
complements on every use to observe updated original values. No quantum work,
measurements or RNG are added. Scope snapshots in the local binding remain a
separate classical resource. Existing positive-only encodings stay unchanged.

For HMR, require matching basis/measurement guards and clear feedback using
those old guards plus the result. A negative guard that aliases the measurement
result needs an explicit old-value copy in the original Core program; an
unmatched sequence is not silently fused. The artifact/codec vocabulary and
public API stay unchanged. Typed ProgramIR result merges and general float
phase reconstruction remain separate boundaries.

## Local acceptance

The dedicated predicate module owns canonical conjunction admission, complement
slot binding and its scratch-width declaration. Classical parity lowering and
reversible quantum/effect lowering consume the same binding. Positives emit
no prefix and reserve no complement slots. Negative complements are always
computed outside the instruction's scopes and observe the latest original
record when the block is reused.

240 native parity configurations cover every three-bit polarity word, target
alias, parity-source pattern and literal, across all eight input records
(1,920 distinct truth cases). Thirty-six native quantum configurations cover
X/CX/CCX, SWAP/CSWAP, Z/CZ/CCZ and exact NEG on the complete three-qubit basis
and both classical input bits; outputs and Fraction phases match independent
maps. Complex-vector tests include entangled data and all negative conjunction
patterns. Every small guarded HMR/reset branch matches, including a copied
old guard that aliases the measurement output logically and a later R that
must recompute a complement from the new result. No inactive branch is summed
coherently or counted twice.

A mixed example with original width five and successive two/one/one negative
tests emits four complement computations but uses only two additional native
slots. The original unowned holes stay borrowed. Each complement computation
emits three classical primitives; it adds no quantum operands, T synthesis,
measurements or RNG. Native complement slots, local saved-mask slots and
logical caller output slots are separate resource categories.

Run `python -m examples.algorithms.negative_classical_predicates` for the
same `circuit.append(op)` consumer before and after conversion. It measures
two data qubits, performs aliased parity under negative conditions and phases
an entangled target using the updated record. Seeded states, original classical
outputs and measurement count remain equal; the example reports two reused
native complement slots rather than allocating quantum work.

Acceptance: 314 new tests. Concentrated regression passes 1,353 tests with
four optional Qiskit/Aer skips; a disjoint finite-field regression passes 366
without skips, for 1,719 passed total. Strict mypy passes all four touched
production/example files. Black, Ruff, compiler scope, architecture, diff and
public API migration baseline checks pass. No reference source or source
fixture is changed and no API snapshot is regenerated.

Disjunctive reverse predicates, typed ProgramIR conditional result merges and
float-to-exact source phases remain separate open items. Unmatched negative
measurement sequences still fail closed. This is not a G2 completion claim.
