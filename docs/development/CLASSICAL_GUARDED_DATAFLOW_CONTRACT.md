# Guarded classical dataflow integration contract

G2-KCFB01 extends the internal, unreleased classical XOR contract to supply
locked Kickmix `gen_unlookup`'s measured phase-table updates. The user has
explicitly authorized completing missing shared register capabilities and source
call chains in integration PR #1084. This addendum is scoped to internal
semantics; it adds no Stable Core export, public signature, Instruction field,
ProgramIR opcode, or serialized public schema and changes no API snapshots.

## Ownership and required replacement

Core owns assignment parsing, canonical predicate validation and def-use rules.
Compiler owns scope snapshots, temporary result slots and source KMX binding.
Runtime owns scalar/batched execution. Simulation owns density branch updates.
Each consumes Core contracts without importing another implementation domain.
The independent truth-table fake and unchanged native source executor tests
land with this contract before implementation consumers. No second production
classical interpreter or quantum-assisted classical AND is introduced.

## Semantics

A `classical_xor` instruction may additionally carry one existing canonical
`conditions` conjunction of defined classical bits. The assignment reads the
old source values, evaluates its predicate before writing, and updates only
active destinations. An inactive assignment preserves its already defined
old destination. All sources, guard slots and conditional destinations must
have prior definitions, even if a guard is statically false. A destination
may alias a source or guard: all reads are from the old record. Repeated
sources still cancel by parity. Empty conjunctions, duplicate guard slots,
disjunction metadata and other extra keys remain rejected.

Unconditional XOR spelling and behavior are unchanged. Guarded assignments
consume no quantum wires, measurements, noise or RNG. Scalar, batched and
density execution must agree; inactive rows are never overwritten. Runtime
true-bit audit values follow the same observed active mask as the existing
scalar executor. Textual OpenQASM uses its existing conditional-assignment
spelling; restricted provider qualification remains refused.

A source `BIT_INVERT` inside a saved positive condition scope lowers to a
Core XOR with the snapshot slots as its predicate. An inline bit operand
remains a parity input, preserving the previous unscoped binding. Each
`PUSH_CONDITION` snapshots its guard unconditionally, even in an inactive
outer scope. Rewriting a live guard must not alter the saved scope mask.
In-scope BIT_INVERT requires a previously defined destination. Scoped
BIT_STORE0 remains outside the target subset: initialize outside the scope.

Core-to-KMX lowering may reject conditional XOR explicitly until it can
preserve aliasing of an assignment's guard and multiple writes during parity
expansion. Typed ProgramIR similarly refuses conditional XOR until it models
the conditional result merge. Neither route may silently drop the predicate
or declare unsupported branch semantics typed. Ordinary unconditional XOR
and all existing protected behavior remain unchanged.

Acceptance includes exhaustive alias/truth tables, unchanged native nested
scope semantics, both local trajectory strategies, density ensembles, guard
lifetime snapshots, resource/no-RNG invariants, unchanged unguarded encodings,
and negative boundary tests. The unlookup consumer remains a separate module.

## Local acceptance

The contract and independent fake landed first in `b39b12e1`. The completed
consumer tests exhaust source/destination/guard aliases, repeated parity inputs,
positive and negative predicates, inactive preservation, nested snapshots and
live-guard rewriting. Locked native KMX scope execution agrees with the fake.
Both local trajectory strategies and density ensembles agree with the declared
semantics. Invalid reads and conditional destinations are rejected before
execution, including inactive density branches.

The runnable example is
`python -m examples.algorithms.guarded_classical_parity`. It accumulates
`measurement AND (table_bit_a XOR table_bit_b)` through `circuit.append(op)`
for every table input pair, with one data qubit and no quantum scratch.
Measurement count stays one per shot. Seed-identical controls prove the
classical work does not consume RNG or alter the quantum state. Snapshot work
uses one classical slot per active scope depth; result and table slots are
caller-owned and reported separately from quantum work. No T synthesis or
analytical quantum cost is inferred from a classical assignment count.

72 contract/consumer tests pass. The integration regression passes 617 tests
with four optional Qiskit/Aer skips; a disjoint finite-field regression passes
366 tests without skips. Total: 983 passed, four optional skips. Strict mypy
passes all ten touched production/example files; Black, Ruff, architecture,
integration scope and the public API migration baseline pass. Existing
unconditional XOR/KMX, exact phases, arithmetic measurement, lookup and field
erasure consumers remain covered. No public API snapshot was regenerated.

Core-to-KMX conditional assignment lowering and typed ProgramIR conditional
result merge remain explicitly unsupported and open, with negative tests.
Scoped initialization and disjunctions on classical assignments remain refused.
The next source consumer is `gen_unlookup`, including phase-table slot ownership
and complete small coherent branch checks; it is not marked accepted here.


The G2-KCFB02 addendum in [atomic reversible lowering](REVERSIBLE_CLASSICAL_LOWERING.md)
admits Core-to-KMX conditional XOR for one positive conjunction, including
destination/guard aliases. Negative predicates and typed ProgramIR result
merges remain explicitly unsupported. The earlier broad reverse-route refusal
above records the G2-KCFB01 baseline, not the current positive-predicate boundary.


G2-KCFB03 [negative predicate binding](REVERSIBLE_NEGATIVE_PREDICATES.md)
now admits mixed zero/one conjunctions through explicit owned classical
complements. Earlier negative-predicate refusals record the prior baseline.
Disjunctions, typed conditional result merges and general float phase recovery
remain explicit boundaries; positive encoding and public schemas are unchanged.
