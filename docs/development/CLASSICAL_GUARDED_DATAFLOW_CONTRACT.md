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
