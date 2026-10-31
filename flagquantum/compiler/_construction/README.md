# Coherent integer construction profile

Compiler owns this private profile of the existing `_hybrid.model` records.
It verifies coherent integer expression definitions and physical binding
preconditions. Binding validation checks the complete definition graph before
checking physical ports; an unchecked function cannot bypass profile verification.
Arithmetic lowering emits existing Core gates through a shared pure arithmetic
kernel. `frontend_python.py` captures a restricted Python integer expression
without running its body. This package does not run circuits or export a public
`synth` API. Its private CS06-B selector checks the two existing schedules.

The same `IRType`, `ValueId`, `Value`, `Operation`, `Block`, `Region`, source
location and `HybridProgram` are reused. `quantum.uint(width)` represents a
logical basis-encoded value, not an allocated register or a consumed effect
token. Function inputs are immutable basis borrows. Repeated reads of a value
are legal; overlapping physical input/output/workspace ports are not.

Supported records are `arith.constant` with classical INDEX type and a checked
nonnegative integer, `qint.add`, `qint.multiply`, `qint.equal`, `qint.less`, `qint.call`, and `qint.return`.
Functions return one quantum integer expression. Definitions form an acyclic
call graph. This is a closed verifier profile, not an extension to the existing
classical hybrid verifier or its lowering pipeline.

Unknown operations and attributes fail before effect-specific validation. Role
and trust flags cannot silently change the borrowed-input contract. General
qfunc/qperm effects, fresh assignment, truncating in-place XOR, wrapping addition,
bind, free/drop and within-apply need separate semantics; their names are not
aliases for the full-width arithmetic evaluator.

The lowering contract requires `z ^= f(inputs)` for every target bitstring
with zero workspace, preserving phases and cleaning workspace. The verifier
checks definitions and bindings; arithmetic lowering implements this permutation
for constants, addition, multiplication, one-bit equality/less-than predicates,
calls and returns. `fresh_zero` is an allocator obligation; `caller_asserted_zero` is a
caller precondition. Neither is inferred from an external state.

For a small change, edit the corresponding verifier, add an accepted/rejected
scenario, and run:

```bash
python -m pytest tests/team/compiler/test_construction_profile.py -q
python -m pytest tests/team/compiler/test_construction_semantics.py -q
python -m pytest tests/team/compiler/test_construction_dependencies.py -q
python -m pytest tests/integration/test_construction_execution_boundary.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

The CS-01 boundary probes use existing Circuit/IR/runtime capabilities. They
establish destination semantics independently of arithmetic lowering. Three-bit states
pin the physical bit-order boundary. A Bell subsystem probe shows why omitting
an output does not reclaim its entangled partner; it does not implement drop.
Arithmetic lowering requires explicit zero-workspace maps and operation budgets.
The default `retain` schedule retains results and reuses only locally cleaned
multiplier scratch. Addition borrows the first operand, which its kernel restores.
It returns existing CircuitIR for Circuit composition and
execution, including nonzero XOR targets. See the
[CS-03 delivery record](../../../docs/roadmap/CS03_MULTIPLICATION_DELIVERY.md).

See the [CS-01 contract](../../../docs/api-changes/FQ-SYNTH-CS01-20261029.md)
and [Classiq review](../../../docs/roadmap/CLASSIQ_SEMANTICS_REVIEW_20261029.md).

For Python capture, run `tests/team/compiler/test_construction_python.py`.
The [CS-04 workflow](../../../docs/roadmap/CS04_PYTHON_TRIAL_DELIVERY.md)
composes this frontend from Services; Compiler still has no Circuit or Runtime
dependency. Both ordinary functions and optional `qfunc` declarations lower to
the existing verified construction profile, with source diagnostics.

`lifetimes.py` records direct use sites and inverse dependencies after inlining
calls into existing SSA. `lowering.py` uses one emitter for `retain` and
`recompute`. The latter can clean operands between uses, restore missing
dependencies before inverse computation and rebind regenerated kernels to their
current qubits. Only inverse-cleaned allocations return to the zero pool.
This is a deterministic heuristic, not a general free/drop operation or optimal
pebbling. The private `schedule_arithmetic` diagnostic includes peak workspace
and compute/cleanup events; `lower_arithmetic` still returns CircuitIR.

Run `python -m examples.synthesis_workspace`, then
`tests/team/compiler/test_construction_scheduling.py` and
`tests/integration/test_construction_scheduling_execution.py` for the shortest
CS-05 contributor path. The [delivery record](../../../docs/roadmap/CS05_WORKSPACE_SCHEDULING_DELIVERY.md)
states measured costs and separates large structural checks from small real CPU
phase tests. The [private boundary](../../../docs/api-changes/FQ-SYNTH-CS05-20261029.md)
keeps automatic candidate selection and broader effects outside this change.

CS06-A adds `constraints.CircuitLimits`, the pure shared authority for final
whole-program width, operation and depth ceilings. Its violations reuse existing
diagnostics and ResourceEstimate; it does not select candidates, plan execution
or infer workspace ownership from CoreIR. Services consumes this record when
checking a sealed Runtime plan. See the
[CS06-A contract](../../../docs/api-changes/FQ-SYNTH-CS06-A-20261029.md).

CS06-B `selection.py` enumerates the existing retain/recompute schedules, keeps
their actual allocation histories, and ranks fully checked candidates. Final
CircuitLimits and workspace ceilings are distinct from ConstructionBudget.
`target.py` reuses basis conversion, compilation, routing, Layout and ASAP
scheduling; it predicts normal Runtime instructions without importing Runtime.
Each target stage retains its existing CoreIR content identity, and `validate()`
rejects later mutations without compiling. Runtime confirmation still goes
through CS06-A; C owns the single normal facade and named reading integration.

Run `python -m examples.synthesis_selection` for the real constraint-driven
phase/cleanup workflow, or `python -m examples.synthesis_selection_costs` for the
complete static table and current manual comparison. Relevant tests are
`tests/team/compiler/test_construction_selection.py` and
`tests/integration/test_construction_selection_execution.py`. See the
[contract](../../../docs/api-changes/FQ-SYNTH-CS06-B-20261029.md) and
[delivery record](../../../docs/roadmap/CS06_CANDIDATE_SELECTION_DELIVERY.md).


## Fresh register composition

`register_profile.py` verifies a separate, private profile in the same
HybridProgram model. `quantum.register(width)` denotes live storage;
`quantum.uint(width)` still denotes an arithmetic expression. One linear
QUANTUM_EFFECT argument orders operations. `qreg.declare` creates an uninitialized
named output reference, `qreg.allocate` initializes it to zero, and
`qreg.initialize` initializes it from a fitting unsigned literal. These require
fresh storage. `qreg.gate` admits X/H/S/Z on a checked numeric bit index, where
index zero has weight one. `qreg.compute` initializes a fresh destination from
one existing verified expression function. `qreg.return` returns every declared
output initialized, once and in declaration order. Roles, extra attributes,
unknown operations and nested regions are rejected.

`register_lowering.py` lowers that profile through the existing arithmetic
scheduler. It cleans each arithmetic segment before subsequent register mutation
and reuses only the resulting zero workspace tail. All output registers remain
live. Repeated actual arguments use a qint.call wrapper with unique physical
entry borrows; they are not cloned or bound to overlapping formal entry ports.
The new private RegisterCandidate retains actual ArithmeticCandidate segment
histories because a multi-output program has no single expression lifetime graph.
It adds no serializer, executor or second validation authority.

`select_register_program` and the existing `select_arithmetic` share one target,
constraint, ranking and recovery consumer. Whole-register-program costs include
initialization, ordered preparation and all arithmetic cleanup. The original
Python arithmetic product retains its separate preparation/resource semantics.
The fresh-register circuit assumes all-zero entry storage; this is an allocator
precondition, not a proof about an externally supplied state.

For a small change, edit the corresponding profile/lowering rule and run:

```bash
python -m pytest tests/team/compiler/test_construction_registers.py tests/integration/test_construction_register_execution.py -q
python -m pytest tests/team/compiler/test_construction_selection.py tests/integration/test_synthesis_user_workflow.py -q
```

`tests/register_program_cases.py` is a test-only model producer. It is not a
public builder or a Qmod parser. The current native Python synth API is unchanged;
external native syntax and Services admission remain later milestones. See the
[CS07-B delivery](../../../docs/roadmap/CS07_REGISTER_PROGRAM_DELIVERY.md).

Python single == and < expressions produce quantum.uint(1) with conservative
[0,1] range, even for repeated operands or a specific basis preparation. Both
classical operands fold exactly; otherwise the shared comparison kernel uses
full zero-extended unsigned operands. bool return annotations require one-bit
results; parameters remain int. The existing lifetime analysis and both
schedules own comparison results and locally clean prefix flags/scratch. The
shared kernel also implements Algorithms' existing greater-than comparator.
See the [comparison contract](../../../docs/api-changes/FQ-SYNTH-COMPARISONS-20261031.md)
and [delivery](../../../docs/roadmap/SYNTH_COMPARISON_DELIVERY.md). Qmod grammar,
Boolean control flow and effectful quantum blocks are unchanged.

Private output_mode="phase" lowering has no output binding and requires a
one-bit return. It marks the available predicate and literally reverses the
selected X/CX/CCX prefix, including inverse allocation events. Preparation is
outside this U. Native logical targets are verified; conservative PHASE_BASIS
line probes preserve phase compensation, while the existing equality-line
SWAP-witness refusal remains active. The public synthesize keyword is proposed,
not implemented. See [phase construction](../../../docs/roadmap/SYNTH_PHASE_ORACLE_DELIVERY.md)
for exact contracts, successful and refused profiles, and the shortest commands.

## One-bit Boolean expressions

The Python frontend admits and/or/not over one-bit unsigned expression values
and numeric 0/1 constants. It validates all operands, including source that a
host Python expression could short-circuit; source bodies are never executed.
Multi-bit truthiness is rejected at the offending operand's source location.
Boolean expressions compute coherent values; classical if/loops, side effects,
bool parameters/literals and bitwise operators keep their existing refusals.

Reuse the existing graph: AND is qint.multiply, NOT is qint.equal with zero,
and OR is their De Morgan composition. No Boolean opcode, type, kernel or
allocation scheme is added. Existing verifier/ranges/lifetimes, both workspace
schedules, value XOR lowering and phase prefix-adjoint cleanup own the resulting
circuit. Definition and expanded-operation budgets apply to every generated
operation. Value and phase target boundaries are unchanged.

Run python -m examples.synthesis_boolean_predicate for the public workflow.
For changes, start with tests/team/compiler/test_construction_booleans.py, then
run tests/integration/test_synthesis_boolean_predicates.py. The latter verifies
complete truth tables including nonzero XOR destinations, complex amplitudes
entangled with an external reference, full-size input pairs, both public modes,
helpers and cleanup. See the [semantic contract](../../../docs/api-changes/FQ-SYNTH-BOOLEAN-PREDICATES-20261031.md).

## Direct controlled-predicate phase markers

For a terminal returned product of two quantum.uint(1) values in phase mode,
keep both operand encodings live and mark with CZ, reducing identical encodings
to Z. The final product's result and scratch are omitted. Existing lifetime,
allocation, source and resource records remain authoritative; a discarded
product has no schedule event. Other phase returns and all value XOR stay on
the existing path. No new control type, opcode or source syntax is required.

Capture the actual forward prefix after both operands have been ensured, then
invert that same prefix after the marker. Recompute may restore dependencies;
neither operand may be erased before marking. Preparation stays outside the
oracle. This is controlled predicate phase, not generic control of an arbitrary
unitary block. Public phase target admission stays logical CPU only.

Run python -m examples.synthesis_controlled_predicate. The Compiler unit tests
compare actual trace/operation boundaries and an independent general prefix
built from the unchanged value evaluator. Integration tests compare full states
including control-zero references, true/false predicates, aliases, entanglement,
interference, dirty-workspace probability and public resource ceilings. See the
[controlled-predicate contract](../../../docs/api-changes/FQ-SYNTH-CONTROLLED-PREDICATE-20261031.md).

## Static range loops and carried SSA values

Python for statements use a fresh scoped index and builtin range with 1-3
nonnegative classical integer arguments and a positive step. Iteration counts
must resolve from constants, never quantum state. A capture expands at most
256 iterations across helpers and sequential loops, with one active loop at a
time. Existing AST/operation/width/helper budgets also apply.

Locals initialized before the loop can be updated once per iteration; every
update is a fresh SSA value. Input parameters and the induction variable remain
read-only. Per-iteration temporary names are single-assignment and do not
escape. Prior aliases keep their original value; outer assignments retain the
existing single-binding rule. No runtime loop or new IR/schema is introduced.

Empty ranges keep carried bindings unchanged. Their body is still checked once
in an isolated environment; provisional operations and value IDs are rolled
back so unsupported hidden calls cannot escape validation. Source helper names
cannot be shadowed by numerical bindings. Structural/nesting/scope errors occur
before lowering or planning; source body execution is never a fallback.

Run python -m examples.synthesis_static_loop. Start changes with
`tests/team/compiler/test_construction_static_loops.py` and then
`tests/integration/test_synthesis_static_loops.py`. Acceptance compares complete
looped/explicitly expanded computations, arbitrary XOR targets, complex states,
external references, value/phase paths and clean workspace. Source operation
counts or identical gate sequences are not substitutes for that equivalence.
See the [loop contract](../../../docs/api-changes/FQ-SYNTH-STATIC-LOOPS-20261031.md).

At capture completion, unused arith.constant definitions are omitted. Their
compile-time values do not consume physical binding width or obstruct a terminal
phase product. Quantum operations are retained. Value IDs may have gaps;
lifetime analysis uses its own inlined program IDs, not original capture IDs.

## Coherent conditional values

The Python expression `yes if predicate else no` captures one private
`qint.select` in the existing HybridProgram. The predicate is one quantum bit;
classical 0/1 folds after validating both expressions. Ranges use the branch
maximum, preserving one-bit predicate/phase composition and avoiding the extra
bit from adding mutually exclusive products. Lifetimes retain all quantum
operand identities for exact inverse computation, including repeated aliases.

The existing arithmetic emitter zero-extends both branches and computes
`no XOR (predicate AND no) XOR (predicate AND yes)` into fresh result storage
with X/CX/CCX and no extra kernel scratch. Repeated controls simplify, without
flipping borrowed inputs. Retain/recompute and exact phase-prefix inversion
remain authoritative. No public type, executor or generic controlled-block API
is introduced; statement-level if/else remains unsupported.

Run `python -m examples.synthesis_conditional`, then
`tests/team/compiler/test_construction_selection_expression.py` and
`tests/integration/test_synthesis_conditional.py`. See the
[contract](../../../docs/api-changes/FQ-SYNTH-CONDITIONAL-EXPRESSION-20261031.md)
and [delivery](../../../docs/roadmap/SYNTH_CONDITIONAL_EXPRESSION_DELIVERY.md).

## Unsigned comparison chains

The Python frontend accepts ==/!=/</<=/>/>= and adjacent chains such as
`low <= x <= high`. It normalizes to existing equal/less, one-bit negate and
conjunction. Both Boolean composition and chained comparisons use the same
conjunction emitter. Each original operand is captured once, left to right;
normalization swaps only the current pair and keeps the original middle value
for the following comparison. All operands validate even after a false prefix.

The existing IR, verifier/lifetimes, kernels, both schedules, precise phase
inverse and Program path do not change. This is pure value computation, not
Python short-circuit execution, identity/membership or statement-level control.
The prior single-comparison refusal cases now have positive full-domain checks;
false-prefix refusal is preserved using an unsupported identity test.

Run `python -m examples.synthesis_interval`, then
`tests/team/compiler/test_construction_comparison_chains.py` and
`tests/integration/test_synthesis_comparison_chains.py`. See the
[contract](../../../docs/api-changes/FQ-SYNTH-COMPARISON-CHAINS-20261031.md) and
[delivery](../../../docs/roadmap/SYNTH_COMPARISON_CHAINS_DELIVERY.md).
