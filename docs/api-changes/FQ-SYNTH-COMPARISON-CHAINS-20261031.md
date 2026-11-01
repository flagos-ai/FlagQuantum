# Full unsigned comparisons and chained predicates

Date: 2026-10-31. The human instructed merging #981 and continuing the sixth
plan. Its exact head 30c68501f1b070ff13bde4a20332284ed2175237 passed 57
checks with three configured skips and merged as
7639db0c477bbb0a9350aa22cf7310276f0797e4. This stage-2 slice completes ordinary
unsigned comparisons and chained interval predicates behind the existing entry.
Public signatures, schemas, contracts and snapshots do not change.

## Source semantics

Accept ==, !=, <, <=, > and >= on existing unsigned quantum expressions and
nonnegative classical integers. Unequal widths zero-extend and do not truncate.
Results are one-bit integers, not consuming references or host Python booleans.
Identity and membership operators, bool literals/parameters, multi-bit implicit
truthiness, signed/fixed-point values and statement-level control remain refused.

A chain such as `low <= x <= high != excluded` is the conjunction of adjacent
comparisons. Each source operand is captured exactly once in left-to-right order;
a middle helper or arithmetic expression is shared between its two comparisons.
The previous original operand, not a normalized/swapped comparison operand, is
carried to the next pair. All operands and operators are validated, even after
a classically false pair. This declarative capture never short-circuits a host
call, measures an input, executes user Python or hides unsupported source in an
unselected portion. All-classical comparisons and conjunctions fold exactly.
Existing AST/expression/helper/expanded-operation/width budgets remain in force.

Predicates compose with and/or/not, conditional expressions, registered helpers,
optional qfunc and static loop assignments. Existing value mode uses XOR output;
phase mode marks (-1)**predicate with the same exact cleanup and target gates.
No general quantum control/within-apply or native Qmod grammar is added.

## Reuse

Normalize != to not equal, > to swapped less, <= to not swapped less and >=
to not less. Reuse the existing one-bit negate and conjunction construction
for chains and Boolean expressions. Emit only existing qint.equal/qint.less/
qint.multiply and constants. Existing HybridProgram/SSA, verifier/ranges,
lifetimes, arithmetic kernels, retain/recompute, exact phase-prefix inverse,
finite target/resource selection, Program/run and named reading remain
unchanged. No new opcode, type, region, serializer, executor or dependency.
Only the Compiler Python frontend changes production behavior.

## Acceptance

Check every relation against independent Python integer truth over complete
small domains and nonzero XOR targets, both schedules, mixed widths, classical
operands on both sides, aliases and boundary values. Verify mixed-direction
chains to detect swapped-middle bugs, helper capture once, false-prefix errors,
source positions, constant folding and expansion budgets. Preserve existing
unsupported identity/membership/Boolean/source refusals.

Compare full complex states, coherent targets and an external entangled
reference without discarding global phase. Validate public numeric/phase
programs, all interval input values, known basis/uniform/complex preparation,
line target conversion where admitted, both objectives, constraints, memory,
no planning after source errors and a complete gate-free CLI example. Record
actual resources. Keep original-size Qmod regression, the 3/3 original-file
milestone, 0/2 independent trials and release_authorized=false separate from
this Python grammar extension.
