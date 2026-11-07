# Disjunctive quantum predicate binding

G2-KCFB04 admits the existing Core quantum `condition_clauses` vocabulary in
reversible lowering. Core canonicality/def-use checks still run first. This
adds no Core fields, opcodes, exports or public schema; classical XOR and
guarded measurement/reset retain their existing conjunction-only grammar.

For multiple clauses, precompute each distinct negative literal in owned
complement slots without modifying any original bit. Initialize an owned OR
accumulator to zero. For each clause, compute NOT accumulator into another
owned slot, push that mask and every positive/complemented clause literal,
then invert the accumulator once and pop the scopes. A clause can set a false
accumulator, but cannot toggle an already true accumulator. Finally emit one
quantum gate under the accumulated positive mask. Overlapping true clauses
never cause repeated quantum application or phase cancellation.

Clauses execute in their original canonical order; invalid duplicated,
subsumed, unsorted or undefined conditions are refused, not normalized away.
All predicate writes stay above the complete original classical namespace,
including unowned holes. Distinct negative literals are shared within the
predicate. Declared native scratch width is unique negative literals plus two
for a disjunction, and remains the maximum requirement across instructions.
Local scope-copy scratch is a separate resource. There is no new quantum work,
measurement or RNG. Single conjunction code remains unchanged.

This compiler adaptation is checked against the unchanged locked native
BIT_STORE0/BIT_INVERT/PUSH/POP executor and independent Boolean/gate maps.
It does not invent a new native opcode or claim source text equality between
the Core Boolean expression and its explicit classical implementation.
Typed ProgramIR conditional result merge remains a separate G2 task.

## Resources and local acceptance

For `k` clauses, `r` distinct negative input literals and clause lengths `L_i`,
the prefix emits `3r + 1 + sum(6 + 2L_i)` classical native primitives. It uses
`r + 2` declared classical scratch slots and emits the conditioned quantum
operation once. Nested saved-mask slots on the local execution route are
reported separately. Classical scratch is owned temporary state, not borrowed
quantum work required to return to zero. No additional quantum wires or quantum
instructions are used to evaluate OR; the measured work registers still clear
in each supported branch.

The untouched locked native executor passes all 253 minimal three-bit clause
pairs, including overlapping and tautological predicates. Nine quantum gate
families across eight predicate families pass native basis and exact fractional
phase checks, as well as arbitrary complex-vector comparison against the direct
Core route. Eight predicate families additionally pass all eight forced HMR
branches on entangled complex input, preserving absolute amplitudes and clearing
all three measured work qubits. An intervening in-place classical inversion
checks that a subsequent use recomputes the mask. The namespace-hole test keeps
caller-owned slots below the original maximum out of scratch allocation.

Run `python -m examples.algorithms.disjunctive_classical_predicates` for the
`circuit.append(op)` example. Both trajectory and batched execution retain the
same original classical outputs, complex final states and measurement counts
through artifact/KMX round trip.

There are 162 new tests. The concentrated regression passes 1,506 tests with
four optional Qiskit/Aer skips; the added branch/hole cases pass with the full
162-test predicate file. The disjoint finite-field regression passes 366 tests:
1,881 distinct passing tests in total. Black, Ruff, strict mypy on the binder
and example, compiler scope, architecture and public API baseline checks pass.
These are measured test outcomes and native primitive accounting, not a
quantum resource model or a complete G2 acceptance claim.
