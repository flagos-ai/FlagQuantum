# Construction compiler

Python qfunc and adapted Qmod programs share the existing HybridProgram model.
Compiler owns semantic verification, lifetime analysis, constrained candidate
selection and lowering to CircuitIR.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `frontend_program.py` | Capture source-retaining Python qfunc programs |
| `frontend_python.py` | Capture the supported scalar Python expression profile |
| `register_profile.py` | Register types, operation attribute admission, preparation and publication shapes |
| `register_verifier.py` | Ordered effects, initialized storage, ownership, aliases and lifetime checks |
| `register_expansion.py` | Expand verified static register calls and inverse composition |
| `register_lowering.py` | Lower admitted register effects to existing gate kernels |
| `lifetimes.py`, `lowering.py` | Arithmetic use analysis, computation and reversible workspace scheduling |
| `selection.py`, `target.py` | Evaluate supported schedules and verify target constraints |

`verify_operation_attributes` checks the operation's exact permitted attributes
before the body verifier inspects operands or modifies its local storage state.
The body verifier remains the single authority for effects and lifecycle. This
ordering preserves the first rejection when malformed attributes and invalid
storage use occur together.

For a small change, start with the relevant accepted and rejected scenarios in
`tests/team/compiler/test_construction_registers.py` and the end-to-end
`tests/integration/test_synthesis_python_program.py`. The [synthesis examples](../../../examples/synthesis/README.md)
provide runnable public programs and independent Circuit comparisons.

## Profile development notes

The following notes retain the semantics and delivery context of the successive
construction profiles.

### Coherent integer construction profile

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
nonnegative integer, `qint.add`, `qint.multiply`, `qint.equal`, `qint.less`, `qint.call`, and `qint.return`. The Qmod adapter also uses
`qint.subtract_mod` for raw word subtraction modulo the declared result width.
Qmod owns sign extension and binary-point alignment before this operation.
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

Run `python -m examples.synthesis.tutorials.workspace.workspace.main`, then
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

Run `python -m examples.synthesis.tutorials.workspace.selection.main` for the real constraint-driven
phase/cleanup workflow, or `python -m examples.synthesis.tutorials.workspace.selection_costs.main` for the
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

Run python -m examples.synthesis.tutorials.arithmetic.boolean_predicate.main for the public workflow.
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

Run python -m examples.synthesis.tutorials.control.controlled_predicate.main. The Compiler unit tests
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

Run python -m examples.synthesis.tutorials.composition.static_loop.main. Start changes with
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

Run `python -m examples.synthesis.tutorials.arithmetic.conditional.main`, then
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

Run `python -m examples.synthesis.tutorials.arithmetic.interval.main`, then
`tests/team/compiler/test_construction_comparison_chains.py` and
`tests/integration/test_synthesis_comparison_chains.py`. See the
[contract](../../../docs/api-changes/FQ-SYNTH-COMPARISON-CHAINS-20261031.md) and
[delivery](../../../docs/roadmap/SYNTH_COMPARISON_CHAINS_DELIVERY.md).

## Ordered phase borrows

`qreg.phase(effect, *registers)` names an existing verified one-bit expression
callee. It borrows initialized references, preserves basis values and returns
the next effect without initializing or consuming a reference. Phase may change;
this is a basis-constant quantum borrow, not a classical constant or a claim
that the quantum state is unchanged. Extra Const/trust annotations are refused.

The same register call checker and unique-actual wrapper serve compute and
phase. The arithmetic scheduler emits its selected predicate prefix, marker and
exact inverse before the following register action can mutate an input. Only
that proven-zero internal tail is reused; all declared outputs remain live.
This is one audited call boundary, not general within/apply or free/drop.

The same phase-target guard protects arithmetic and register producers. The
contributor fixture works on native logical targets and unoptimized PHASE_BASIS
routing with physical layout retained. Its restored route is refused because
the existing normalizer cancels restore-SWAP witnesses. That strict evidence
check remains, and no unverified mapping is passed to Runtime.

Run `python -m examples.synthesis.tutorials.control.register_phase.main`, then
`tests/team/compiler/test_construction_register_phase.py` and
`tests/integration/test_synthesis_register_phase.py`. The example is a private
owned-IR contributor trial. The public Python predicate path is unchanged;
public syntax/roles and native Qmod admission follow separate review. See the
[contract](../../../docs/api-changes/FQ-SYNTH-REGISTER-PHASE-20261101.md) and
[delivery](../../../docs/roadmap/SYNTH_REGISTER_PHASE_DELIVERY.md).

## Const and fresh-output register calls

Private register callees reuse HybridProgram: block arguments after the effect
are initialized Const references; qreg.declare operations are fresh outputs.
The final return lists all borrows then outputs, once and in order. A Const
preserves computational-basis values while allowing phase and entanglement to
change. It cannot be targeted by a gate or initialization, computed into, or
passed as a nested Output. Structural refusal includes cancelling gate pairs;
arbitrary unitary-equivalence proofs are not part of this profile.

qreg.call passes Const actuals first and fresh Output actuals second. Exact
widths and initialized reads are mandatory, even for unused Const parameters.
Const actuals may alias each other; outputs must be disjoint from every actual.
All definitions validate, including unused callees. Root programs keep the
effect-only entry and all-zero allocation precondition; borrowed callees cannot
be selected as fresh roots. Consuming Input is not admitted by this contract.

register_profile.py verifies roles and the acyclic call graph. The internal
register_expansion.py substitutes verified references and rethreads effects;
register_lowering.py then uses the unchanged arithmetic/phase schedules. The
selector checks reachable expanded phase actions before target conversion, so
nested calls cannot hide phase effects. Limits include 32 call levels, the
existing 4096 source/final model nodes, and 4096 expansion visits including
empty calls. No alternative executor, public type or role manager is needed.

Run `python -m examples.synthesis.tutorials.composition.const_calls.main` for the contributor-only owned-IR
path, then `tests/team/compiler/test_construction_register_calls.py` and
`tests/integration/test_synthesis_const_calls.py`. Public Python stays unchanged.
Ecosystem now delegates native Const/Output helpers through this boundary; its
former Output-only substitution and alias checker have been removed. The new
public source example is `python -m examples.synthesis.tutorials.qmod.qmod_const.main`; source role
translation owns no duplicate Compiler role/effect checker. See
the [contract](../../../docs/api-changes/FQ-SYNTH-CONST-CALLS-20261101.md) and
[delivery](../../../docs/roadmap/SYNTH_CONST_CALLS_DELIVERY.md).

## Computed storage cleanup

Private qreg.uncompute(effect, destination) returns a directly computed result
to fresh proven-zero storage. A structural proof tracks the result and its
direct inputs since qreg.compute. A write to either invalidates that proof,
including a diagonal gate or a cancelling gate pair; compute/phase/Const reads
preserve it. Cleaning a dependency invalidates the dependent result's cleanup
authority. Arbitrary initialization, user zero assertions and helper Outputs
do not create direct-compute authority. Const references cannot be cleaned.

register_lowering.py retains the actual selected ArithmeticCandidate and emits
its exact reverse, with the same physical bindings and cleaned work tail. The
shared lowering adjoint check admits only payload-free X/CX/CCX. Reverse events
exchange compute/uncompute and retain their allocations, scratch and live-width
observations. Actual forward and inverse segments, rematerializations, complete
gate counts, workspace peaks and operation budgets use the existing records.
No second synthesis decision is made to obtain the inverse.

The destination reference and physical slot remain owned. A subsequent compute
or initialization can reuse them, but reads/returns while fresh and double
cleanup fail. All declared outputs still return initialized in order; no local
escape, pool release, Input consumption, drop or free is implied. This is a
closed prerequisite for later local/within scopes, not a general unitary inverse.
Every helper verifies independently, including unused ones, and the expanded
root verifies again through the existing call boundary.

Run `python -m examples.synthesis.tutorials.composition.register_cleanup.main`, then
`tests/team/compiler/test_construction_register_cleanup.py` and
`tests/integration/test_synthesis_register_cleanup.py`. This owned-record example
is for contributors; public Python/native source syntax is unchanged. See the
[contract](../../../docs/api-changes/FQ-SYNTH-COMPUTED-CLEANUP-20261101.md) and
[delivery](../../../docs/roadmap/SYNTH_COMPUTED_REGISTER_CLEANUP_DELIVERY.md).

## Local ownership and zero-slot reuse

Private qreg.local declares fresh temporary storage without adding a named
output. qreg.release(effect, local) requires a live owned local in fresh
proven-zero state, invalidates its reference and makes its slots available to
later locals. A computed local uses the selected qreg.uncompute inverse first.
Allocated/initialized state, even a known zero, cannot simply be released.
Const references, outputs, stale references, trust attributes and local escape
are refused. Every local must be released before return; outputs and borrows
retain their existing initialization/order rules.

register_ports maps named outputs first and assigns locals the lowest available
zero slots by lifetime. Locals may have noncontiguous MSB-first ports. Arithmetic
scratch starts after the high-water local band for the entire program, so a
later local cannot dirty scratch captured by an earlier selected inverse.
Whole workspace is local-band peak plus maximum arithmetic scratch; resource,
qubit, memory and workspace budgets include all hidden storage. Even outputs
declared late remain reserved. This is conservative physical allocation, not
automatic rewriting of source effect order or a global-width optimum.

register_expansion gives each helper local a fresh SSA identity and collision-free
internal label per call. Only Const and declared Output roles cross a call;
locals do not escape. Every helper, including unused definitions, verifies
independently and the expanded root verifies again. Existing model/visit/depth
budgets and exact phase targets remain. No Runtime allocator or public API is
added, and native local/within/free/drop syntax is still outside admission.

Run `python -m examples.synthesis.tutorials.composition.local_registers.main`, then
`tests/team/compiler/test_construction_local_registers.py` and
`tests/integration/test_synthesis_local_registers.py`. The contributor example
returns x/y/reference, with every temporary hidden and zero. See the
[contract](../../../docs/api-changes/FQ-SYNTH-LOCAL-REGISTERS-20261101.md) and
[delivery](../../../docs/roadmap/SYNTH_LOCAL_REGISTERS_DELIVERY.md).

## Python positive literal powers

Python capture admits x**n for literal integer n in 1..8. The base is captured
once; exponent one reuses its value and larger exponents emit at most seven
existing multiplications. One private arithmetic routine owns constant folding,
complete-domain width checks and emission for +, * and powers. The square graph
remains unchanged. No function body is evaluated or power kernel introduced.
Boolean, zero, negative, fractional, configured, computed and larger exponents
are refused. Compiler node/visit, 256-bit and physical resource budgets remain
in force. Run python -m examples.synthesis.tutorials.arithmetic.python_powers.main and the focused
Python power integration suite; this does not establish notebook acceptance.

## Range-proven constant addition

The arithmetic scheduler uses result-only X/CX copies for a positive classical
addition constant whose bits are disjoint from the complete unsigned input
domain mask. Addition then equals XOR. Both operand orders use this proof;
initialization does not narrow the domain. Zero, overlapping constants and two
quantum operands keep ripple lowering. Existing inverse cleanup, retain and
recompute schedules, resource accounting and Program execution remain shared.

Run `python -m examples.synthesis.tutorials.arithmetic.carry_free_add.main` and
`tests/integration/test_synthesis_carry_free_add.py`. The one-bit x+256 example
uses 19 qubits and an 8 MiB planned state buffer. See the
[contract](../../../docs/api-changes/FQ-SYNTH-CARRY-FREE-ADD-20261101.md) and
[delivery](../../../docs/roadmap/SYNTH_CARRY_FREE_ADD_DELIVERY.md).

## Multiplication by powers of two

For a positive-domain quantum operand multiplied by a classical 2**k (k>=1),
the scheduler copies its bits into the high result bits and reserves no generic
multiplication scratch. Low k result bits stay zero; the full result width is
still allocated. Forward, inverse and rematerialized computation read current
bindings. Both operand orders work. Zero, one, other constants, two quantum
operands and derived zero-bound values retain generic multiplication.

Run `python -m examples.synthesis.tutorials.arithmetic.power_two_multiply.main` and
`tests/integration/test_synthesis_power_two_multiply.py`. The one-bit x*256
example uses 19 qubits and an 8 MiB planned state buffer. See the
[contract](../../../docs/api-changes/FQ-SYNTH-POWER-TWO-MULTIPLY-20261101.md) and
[delivery](../../../docs/roadmap/SYNTH_POWER_TWO_MULTIPLY_DELIVERY.md).

## Constant phase in register effects

The existing qreg.gate record admits gate=phase with exactly gate/index/angle,
a finite Python float in radians and an initialized target. It emits the
existing Core phase opcode with named theta; other gates retain gate/index.
Const targets, unknown state and malformed indices/attributes remain refused,
and gate writes retain dependency invalidation. Native source policy is owned
by Ecosystem, which currently admits only integer literals 0..2**53 on qbits.

Run `python -m examples.synthesis.tutorials.qmod.qmod_phase_gate.main`, then
`tests/team/compiler/test_construction_phase_gate.py` and
`tests/integration/test_synthesis_qmod_phase_gate.py`. Full complex amplitudes
are necessary to distinguish phase from RZ's global-phase convention. See the
[contract](../../../docs/api-changes/FQ-SYNTH-CONSTANT-PHASE-GATE-20261102.md).

## Unified program boundary and RZ literals

The [UP-00 contract](../../../docs/api-changes/FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md)
fixes program semantics. UP-01 now admits the typed preparation/arithmetic/call
subset below; scalar qfunc snapshots retain their bounded unsigned profile.
The unreleased float-return radians evaluator is withdrawn. Scalar capture
keeps int/bool annotations; typed programs use explicit Output/Const ports and
None returns. Neither path executes host bodies. General explicit real phase
and coherent control remain follow-on work; no second canonical IR is added.

The closed constant-angle register gate set shares phase/rz validation and Core
named-theta lowering. Source decimal gates are owned by Ecosystem, not the Python
integer frontend. Run `tests/team/compiler/test_construction_rz_gate.py` and
`tests/team/compiler/test_construction_program_boundary.py`; use
`python -m examples.synthesis.tutorials.qmod.native_and_qmod_rz.main` for runnable high-level integer,
Qmod and Circuit comparisons with the remaining native phase gap explicit.

## Python program capture (UP-01)

frontend_program owns the typed AST-to-register handoff. program_types supplies
annotation roles/types and captured operation symbols; those never allocate or
calculate in host Python. frontend_python reuses its source reader and captures
only annotation-name bindings from the lexical definition scope. Body call
bindings resolve statically at synthesis, then the existing checked Program is
independent of later namespace changes. No arbitrary annotations or properties
are evaluated.

The frontend emits existing qreg allocate/initialize/gate/compute/call/return
operations and reuses scalar capture for mathematical expressions. It retains
exact source positions, infers full-domain unsigned widths and bounds static
specializations/calls. Existing register verification owns initialization,
Const/fresh-Output, alias and cleanup proofs; selection/lowering reuse existing
kernels, schedules, targets and resource checks. No Qmod string bridge or second
IR is involved. General inverse/phase/control/local scopes remain later slices.

Golden path: run `python -m examples.synthesis.tutorials.composition.python_program.main`, change a literal
allocation width, then run `tests/team/compiler/test_construction_python_program.py`
and `tests/integration/test_synthesis_python_program.py`. Include full complex
state/clean-workspace checks, not just output probabilities. The typed program
[contract](../../../docs/api-changes/FQ-SYNTH-PYTHON-PROGRAM-UP01-20261102.md)
and [delivery](../../../docs/roadmap/SYNTH_PYTHON_PROGRAM_UP01_DELIVERY.md) record
exact API scope, source guards, original-profile preservation and validation.

## AO-01 arithmetic normalization and bit facts

The shared value scheduler now verifies the original graph, keeps its original
candidate, and normalizes an immutable inlined pure segment in
`arithmetic_rewrites.py`. Canonical operand order uses width and structural
rank, not user names. Constants, zero/one identities, common pure operations
and dead expressions are shared/simplified without narrowing any declared
input or return type. A typed quantum constant witness remains where an INDEX
constant could not legally stand for quantum output storage.

`bit_facts.py` describes complete declared domains: effective width and known
zero/one masks. Addition propagates possible carries; multiplication uses sound
conservative trailing-zero, power-of-two and same-SSA-square facts. This does
not introduce a square kernel or general Boolean-network search. The 4096-node,
256-bit profile bounds apply before rewriting. No input-state narrowing or
cross-effect/global memoization is permitted. `lifetimes.py` retains supplied
source locations and explicit SSA call sites; scalar helpers already inlined by
Python capture retain the helper-body locations that frontend supplied.

The same emitter builds original and rewritten schedules. Actual workspace,
operation count and ASAP depth come from the existing allocation history and
resource estimator. Adopt a rewrite only when none of those logical metrics
increases, or when the original cannot fit the construction budget and the
verified rewrite can. Keep the original SSA, original candidate or its exact
budget diagnostics, masks, square identities, source origins and normalized
costs on private `ArithmeticCandidate` fields. These do not add a public report
schema or another canonical IR. The original construction caps and both existing
schedule checks remain; this is not target-aware Pareto search or optimality.

Each `qreg.compute` gets its own invocation; a quantum write, another register
segment, or region cannot reuse a stale computed value. Original initialization,
alias, effect/version and cleanup validation runs before simplification. Exact
adjoint cleanup retains the matching original candidate's inverse evidence.
The phase-mode emitter is unchanged; phase tests use the explicitly unnormalized
value emitter when constructing their historical general reference.

Run `python -m examples.synthesis.tutorials.arithmetic.normalization.main` for the unified qfunc program,
Qmod and an independently derived Circuit reference. Follow
`tests/team/compiler/test_construction_normalization.py` for a small rule change,
then `tests/integration/test_synthesis_normalization.py` for complete-state,
XOR, resource-guard and mutation-boundary acceptance. The AO-00 JSON and measured
revision stay frozen; new results use separate AO-01 files.

## AO-02 arithmetic implementation family

`schedule_arithmetic` retains the actual AO-01 schedule and probes one bounded
family: symmetric same-SSA squaring, constant shift/add using shared Cuccaro
sweeps, and direct terminal XOR addition. The existing power-of-two and
disjoint-constant proofs are reused. Accumulation kernels only write internal
zero product storage; the terminal writer supports arbitrary nonzero targets.
Its carry prefix excludes output wires and is inverted after XOR writes.

The local guard requires nonincreasing workspace/operations/depth and one strict
gain. `conservative`, `implementation_cost`, `implementation_rules` and refusal
diagnostics keep private evidence; `baseline`/`normalization` retain earlier
representations and source facts. Public candidate rows and budgets remain
unchanged. AO-04 owns target-aware implementation enumeration.

Terminal events have an empty owned allocation because result storage belongs
to the external caller. Their paired compute/write and carry-cleanup events
account for scratch without freeing caller output. Exact adjoints reverse those
actual gates/events. The phase emitter remains conservative.

Run `python -m examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main` for the complete
native qfunc/Qmod workflow. Kernel tests are in
`tests/team/compiler/test_construction_arithmetic_implementations.py`; integration
acceptance covers coherent targets, controls, adjoints and both schedules.
See [the AO-02 delivery record](../../../docs/roadmap/SYNTH_ARITHMETIC_AO02_DELIVERY.md)
for proofs, scope, fixed receipts and the remaining AO-03/04 gap.
