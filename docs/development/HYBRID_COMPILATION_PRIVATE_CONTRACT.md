# Private hybrid compilation contract

Status: Phases 1-24 implemented and verified under the repository owner's
2026-09-09 direction to record and execute the Python-first hybrid compilation
plan.

## Purpose

Authorize the smallest Compiler-owned semantic core needed to represent the
first hybrid quantum-classical vertical slice without changing a stable public
API or creating a second execution contract.

This contract is deliberately narrower than the long-horizon multi-level IR
architecture. It authorizes only the semantic objects that are exercised by the
first nested-loop, data-dependent-gate, scalar-expectation scenario.

## Existing contracts reused

- Core-owned `CircuitIR` remains the quantum-region handoff.
- Existing compiler entry points continue to consume and return `CircuitIR`.
- Existing Runtime entry points continue to consume `CircuitIR` or their
  existing execution-plan type.
- Existing local statevector execution remains the first numerical target.
- Existing tensor and result types remain authoritative at the execution edge.
- Existing extension discovery and lifecycle remain the only plugin authority.

No new public artifact, execution result, backend, plugin registry, or root
export is authorized.

## Newly authorized private responsibility

`flagquantum.compiler._hybrid` may own a private structured program
representation containing:

- typed SSA-like values;
- immutable operations, blocks, and regions;
- function arguments and one return;
- scalar/tensor constants, indexing, comparison, addition, and remainder;
- structured `if` and `for` regions with explicit yields;
- ordered quantum effects;
- parameterized RX, RY, CX, embedding, and scalar expectation operations;
- deterministic semantic identity and source-located diagnostics;
- a fail-closed verifier.

These types are compiler implementation details. They must not be re-exported
from `flagquantum`, added to the Stable Core manifest, serialized as a public
schema, or accepted by `fq.run`/`fq.plan` during Phase 1.

## Dependency direction

The Phase 1 package may depend on Python standard-library modules and stable
Core semantic types needed for later `CircuitIR` lowering. It must not import
Runtime, Simulation, Compute, Remote, Ecosystem, provider SDKs, or optional
frameworks.

PyTorch is not required by the Phase 1 semantic core. Framework tensor values
enter only in later capture and execution phases behind separately tested
adapters.

## Required invariants

The Phase 1 implementation must prove:

1. immutable deterministic representation;
2. definitions dominate uses within the accepted single-function profile;
3. values have one definition and type-consistent uses;
4. branch result arity and types match;
5. loop-carried value arity and types match initialization and yield;
6. ordered quantum effects cannot be duplicated, reused, or dropped;
7. unsupported operation names and attributes fail closed;
8. source locations do not affect semantic identity;
9. no current stable import, signature, default, schema, or behavior changes;
10. the package remains unused by the default compiler and runtime until the
    later vertical-slice gates authorize integration.

## First-profile exclusions

Phase 1 does not authorize:

- Python AST capture;
- `while`, recursion, exceptions, mutation, or general Python semantics;
- measurement-produced control flow;
- stateful simulator sessions;
- lowering or execution;
- differentiation implementation;
- public API or default-path integration;
- MLIR, LLVM, QIR, timing, pulse, provider, or deployment behavior;
- Catalyst or FlagQuantum-authored C++.

## Entry gates

Implementation begins only when:

- the authoritative Integration baseline is identified;
- migration targets current vNext authority rather than restoring a historical
  worktree, package, or layer stack;
- team-scope preflight passes for all proposed implementation and test files.

These gates are satisfied. Phase 1 implementation proceeds in the private
Compiler namespace on the Integration baseline. The historical `_compiler`
tree remains reference evidence only.

## Phase 1 acceptance

- focused construction and negative-verifier tests pass;
- semantic identity is deterministic under repeated construction;
- source relocation does not change identity;
- all declared invalid value/effect/control-flow cases are rejected;
- compiler package boundaries pass the architecture checker;
- default smoke/unit tests pass;
- protected public API comparison reports no hybrid additions;
- the Compiler README contains a ten-minute internal contributor path.

Phase 1 is complete and authorizes Phase 2 capture design review; it does not
automatically authorize capture, execution, gradients, or public claims.

## Phase 2 restricted capture authorization

Phase 2 may add private `capture_source` and `capture_function` entry points
under `flagquantum.compiler._hybrid`. Capture parses Python source with `ast`;
it must not execute the function or evaluate runtime tensor predicates.

The accepted subset is limited to local-name assignment, structured
`if`/`elif`/`else`, `for` over `range`, iteration over a statically ranked tensor
axis, `enumerate` over such an axis, tensor dimension/extraction, scalar
addition/remainder/comparison, and the first-profile quantum operations. A
single scalar expectation return is required.

Local assignments are straight-line temporaries in their current region.
Writing an outer classical binding inside a branch or loop is rejected until a
later contract adds explicit classical region-carried results; such writes must
never be approximated by retaining the pre-region value.

Mutation, `while`, `break`, `continue`, Python `yield`, exceptions, context
managers, comprehensions, generators, imports, global/nonlocal state, arbitrary
calls, host scalar extraction, detach, NumPy conversion, recursion, and
unsupported quantum operations fail closed with source-located diagnostics.

Phase 2 does not authorize specialization, `CircuitIR` lowering, execution,
autodiff, framework graph capture, or default-path/public integration.

Phase 2 acceptance is complete. The captured golden source preserves nested
tensor-axis loops and data-dependent quantum gate selection, and the prohibited
construct suite fails with source-located diagnostics. Phase 3 specialization
and `CircuitIR` lowering remain separately gated.

## Phase 3 specialization and lowering authorization

Phase 3 may interpret a verified `HybridProgram` against one runtime input set
to select structured branches and unroll bounded loops into an ephemeral quantum
trace. The trace is not a restored `QuantumIR` and is not serialized or exposed;
it exists only to lower the selected path into the existing Core-owned
`CircuitIR`.

Parameterized gates lower first to a `CircuitIR` template containing stable
Core `Parameter` slots. Actual scalar tensor views are retained separately by
reference and may be bound into a circuit without `.item()`, detachment, NumPy
conversion, or scalar copying. Angle embedding lowers to one RX operation per
declared wire. Program structure, input signature, selected quantum structure,
and parameter values remain distinct identity dimensions; parameter values
must not affect reusable structure identity.

A bounded LRU structure cache is permitted only when hit, miss, and eviction
are observable in the returned private result. Unsupported input signatures,
non-scalar predicates, invalid loop bounds, invalid wires, and unsupported
dynamic effects fail before Simulation.

Phase 3 does not authorize Runtime integration, numerical execution, VJP,
public/default-path integration, or performance claims.

Phase 3 acceptance is complete: positive, negative, zero, and mixed gate
selection match hand-built circuit structures; late-bound tensor views preserve
their autograd edges; template identity excludes values; bounded cache events
are observable; and the existing Compiler accepts both templates and bound
circuits. That compiler-owned handoff is the input to the separately reviewed
Phase 4 contract below.

## Phase 4 local CPU forward authorization

Phase 4 may complete the forward-only vertical slice by encoding the captured
single expectation in the existing Core `MeasurementNode` representation on the
same `CircuitIR` that already carries the lowered instructions. Each Pauli term
uses the same `fq_output_index`, so the existing `ExecutionResult.expectation()`
accessor returns their sum as one scalar per batch item. The existing
`ObservableNode` terms remain available as semantic circuit annotations; they
do not constitute a second execution request or result type.

The handoff is exactly:

```text
Compiler-owned specialization and lowering
  -> Core-owned CircuitIR with MeasurementNode requests
  -> existing Runtime plan and one-attempt execution
  -> existing Simulation local CPU statevector
  -> existing ExecutionResult.expectation()
```

Runtime must not import `flagquantum.compiler._hybrid`. It accepts only the
Core-owned artifact and remains responsible for planning, execution lifecycle,
measurement dispatch, and result assembly. Simulation remains responsible for
statevector evolution and Pauli expectation numerics. No new executor,
execution result, public export, serialized public schema, or backend-selection
rule is authorized.

Phase 4 is restricted to analytic, forward-only, local CPU statevector
execution with `single_device_fast_path` semantics. It does not claim a VJP,
gradient preservation through the runtime, finite-shot behavior, noise,
accelerator support, distributed execution, performance improvement, or
production support. The existing public/default `fq.run` and `fq.plan` paths
must not import or invoke hybrid capture or specialization when hybrid
compilation is unused.

Phase 4 acceptance requires all golden positive, negative, zero, and mixed
branch patterns to match an explicit hand-built circuit reference, including
the grouped scalar expectation and execution metadata. Compiler specialization
errors must remain compiler errors, while planning/execution failures retain
their existing Runtime or Simulation exception ownership.

Phase 4 acceptance is complete. The compiler-produced `CircuitIR` passes through
the existing compiler, Runtime planner, and local CPU statevector path; all four
golden branch classes match the explicit reference and return one grouped
expectation per batch item. The result reports `local_statevector` and
`single_device_fast_path`. No Runtime, Simulation, root API, public contract, or
default-path implementation changed. Unknown observable axes now fail during
private Program IR verification rather than acquiring an accidental execution
meaning. That forward handoff is the input to the separately reviewed Phase 5
contract below.

## Phase 5 local CPU VJP authorization

Phase 5 may connect the bound `CircuitIR` to the existing Runtime-owned
statevector reverse executor. The accepted observable profile is a sum of one
or more single-wire Pauli-Z terms. Runtime may extend its internal reverse
executor from one Z wire to several Z wires by evaluating the circuit once and
constructing one adjoint seed equal to the sum of the term adjoints. Numerical
gate derivatives and statevector primitives remain Simulation-owned.

The gradient chain must preserve the original PyTorch scalar tensor views used
by lowering. Backward therefore accumulates from the existing private adjoint
result through those views into the original `weights` and `data` tensors;
host scalar conversion, detachment, NumPy conversion, or replacement by copied
leaf tensors is forbidden.

Control-flow derivatives are branchwise. Backward follows only the gates
selected during the corresponding forward specialization; comparison
predicates themselves are not differentiated. Equality at an ordered
comparison threshold is a non-smooth control boundary. Specialization must
record such boundaries and, when smooth gradients are explicitly required,
fail with a Compiler-owned diagnostic before lowering or execution.

Acceptance compares the adjoint VJP against dense PyTorch statevector autograd
and central finite differences away from branch boundaries. Parameter shift is
not an execution fallback. A deterministic optimization check must decrease a
seeded objective over multiple steps. The reverse result summary must identify
the adjoint method, observable profile, selected wires, and local
`single_device_fast_path` semantics.

This authorization does not cover higher-order derivatives, finite-shot or
noisy gradients, general Pauli strings or weighted Hamiltonians, accelerators,
distributed-gradient claims, performance claims, a public hybrid API, or a
change to the default `fq.run`/`fq.plan` path.

Phase 5 acceptance is complete. The existing statevector adjoint executor now
supports a sum of single-wire Z terms using one forward state and one combined
adjoint seed while preserving its original single-wire call form. Positive,
negative, and mixed branch paths match both dense PyTorch statevector autograd
and central finite differences for the original `weights` and `data` tensors.
A six-step deterministic optimization trajectory decreases monotonically.
Ordered-comparison equality boundaries are recorded and fail before lowering
when smooth gradients are required. The result summary exposes
`statevector_adjoint`, the accepted observable profile, selected wires, and
local backward semantics. Those results form the input to the separately
reviewed Phase 6 contract below.

## Phase 6 PyTorch graph-compilation authorization

Phase 6 may wrap one already specialized `CircuitIR` quantum region in a
private functional PyTorch custom operator. The operator receives every scalar
parameter tensor and the deterministic `CircuitIR` JSON as explicit inputs.
The template records the exact parameter order in
`hybrid_parameter_order`; a process-global program registry, opaque integer
handle, implicit mutable cache, or duplicated circuit representation is not
permitted.

Runtime owns the operator because it invokes the existing Runtime statevector
reverse executor. Compiler remains responsible only for specialization,
lowering, and deterministic template construction. The operator must register
a FakeTensor implementation and a first-order Autograd formula. The Autograd
formula calls a separate functional backward custom operator so AOT Autograd
can retain the whole supported region in a `fullgraph=True` graph. Backward
saves the explicit parameter tensors and immutable circuit text, then
rematerializes the existing statevector-adjoint execution context. This is a
correctness-first implementation, not a claim that quantum state replay has
been eliminated.

The supported profile remains analytic local CPU statevector execution, scalar
`float32` or `float64` parameters, and a sum of unit single-wire Pauli-Z terms.
All parameters must have one dtype and device and must exactly match the
template slots. Unsupported devices, shapes, observables, missing slots, or
extra slots fail closed before simulation. Only the selected quantum region is
compiled; this phase does not claim general dynamic Python capture, predicate
derivatives, higher-order gradients, accelerator or distributed execution, or
finite-shot/noisy gradients.

Acceptance requires `torch.library.opcheck`, numerical `gradcheck`, eager versus
compiled forward and first-order-gradient parity, and successful
`torch.compile(..., fullgraph=True)` execution with AOT Autograd. The custom
operator remains private: it is not exported from the package root and does
not alter `fq.run`, `fq.plan`, `fq.Module`, backend selection, or the default
non-hybrid path. Compile and execution timings may be recorded separately, but
no performance benefit is claimed.

Phase 6 acceptance is complete for this bounded profile. The serialized
template carries an explicit parameter order, the functional forward and
backward operators have FakeTensor coverage, `opcheck` and `gradcheck` pass,
and eager/AOT-compiled forward and first-order gradients agree under
`fullgraph=True`. Original tensor views still receive gradients. The backward
path deliberately rematerializes the statevector-adjoint context, and no
performance, public API, accelerator, distributed, or broader control-flow
claim is made.

## Phase 7 dynamic measurement-session authorization

Phase 7 may add a separately bounded measurement-feedback profile without
changing the analytic custom operator from Phase 6. Source capture represents
`qp.measure(wires=...)` as a `quantum.measure` operation that consumes and
returns the linear quantum effect and also produces a boolean SSA value. A
program may use that value directly as an `if` condition and return the
measured bit. This makes the data dependency explicit in Program IR rather
than evaluating a quantum measurement during Compiler specialization.

Compiler lowers both sides of the structured branch into the existing Core
`CircuitIR`: the measurement becomes an existing dynamic `measure`
instruction, and gates in the true and false regions carry conditions for the
same classical bit. Runtime, not Compiler, owns state collapse, the per-shot
classical register, conditional continuation, seeding, trajectory selection,
statistics, and result construction. Simulation continues to own statevector
gate primitives; this phase reuses the existing dynamic measurement
implementation and adds no numerical kernel. The first profile supports only
no-input programs, top-level measurement, direct measurement-bool conditions,
H/X/CX gates, and local CPU `complex64` or `complex128` execution.

The private Runtime entry point validates the compiler marker and all dynamic
metadata before constructing a transient `DynamicCircuit` and calling the
existing dynamic trajectory executor. Every shot starts from an independent
initial state, maintains its own collapsed quantum state and classical
register, follows its measured branch, and contributes one final sample. A
fixed integer seed must reproduce measurement bits, branches, and final
samples. The returned `DynamicExecutionResult.classical_bits` contains the
source-visible measurement values; metadata identifies which classical bit
was returned by the program.

Stochastic gradients are explicitly unsupported and fail closed when any
trainable tensor reaches the session. There is no parameter-shift,
score-function, straight-through, or silent deterministic-gradient fallback.
This authorization also excludes runtime inputs, measurement inside a branch,
arbitrary boolean expressions over measurement results, loops containing
measurements, reset, durable sessions across calls, `torch.compile`, finite-shot
gradient claims, accelerators, distributed execution, remote providers, and
performance claims.

Phase 7 acceptance requires exact `CircuitIR` lowering, observation of both
branches, shot-wise agreement among the measured bit, conditional gates, and
final samples, seeded reproducibility, equivalent reference and batched
trajectory semantics, and negative tests for invalid shot counts, classical
read-before-measurement, conditional measurement, and trainable parameters.
The entry points remain private and do not change the root API or the default
`fq.run`/`fq.plan` path.

Phase 7 acceptance is complete for this bounded profile. The compiler emits a
single Core `CircuitIR`; the existing Runtime dynamic trajectory machinery
performs state collapse and feedback; both reference and batched strategies
preserve the returned measurement semantics; and every excluded gradient or
control case fails closed. This result is a semantic vertical slice, not a
claim of general dynamic-program compilation or production dynamic-QPU
support.

## Phase 8 parameterized bounded-session authorization

Phase 8 may add non-trainable scalar, index, and boolean runtime inputs to the
private dynamic-session lowering path. Tensor inputs remain excluded. Runtime
input shapes and dtypes use the same specialization validation as the analytic
path. A `complex64` circuit accepts declared `float32` scalar inputs and a
`complex128` circuit accepts declared `float64` scalar inputs.

RX and RY are added to the accepted dynamic gate profile. Compiler stores each
rotation value in an explicit ordered binding map and emits only a Core
`Parameter` placeholder into the `CircuitIR` template. Binding preserves the
original scalar object. Parameter values therefore do not enter the template
hash, while changes to gate count, branch selection, loop trip count, or wire
structure produce a different template and require specialization again.

Positive bounded `range` loops may be specialized and unrolled before Runtime
execution. The default cumulative ceiling is 10,000 iterations and callers may
choose a smaller limit. Measurement inside a loop remains unsupported. Static
input predicates may select a branch during specialization; predicates derived
from measurements continue to lower both branches as classical conditions.

Finite-shot dynamic execution remains non-differentiable. Compiler rejects a
runtime input with `requires_grad=True`; Runtime independently rejects any
trainable tensor introduced into a bound or tampered artifact. There is no
parameter-shift, likelihood-ratio, straight-through, surrogate, or detached
gradient result. An unbound template also fails before trajectory execution.

Phase 8 acceptance is complete for this bounded profile. Equal loop structure
with different scalar values has one template identity, binding retains the
source tensor, RX(0) and RX(pi) produce the expected deterministic measurement
and feedback outcomes, and the unroll, binding, dtype, and gradient exclusions
fail closed. The extension remains private and makes no public API, default
path, `torch.compile`, accelerator, distributed, capacity, or performance
claim.

## Phase 9 explicit loop-carried classical-state authorization

Phase 9 may represent rebinding of an existing scalar, index, or bool local in
the direct body of a bounded `for` loop. Capture identifies those names before
building the region and represents each value explicitly in all four places
required by structured SSA: the `scf.for` operands, body block arguments,
`scf.yield` operands, and `scf.for` results. The result values replace the
outer bindings after the loop, so subsequent gates consume the verified
post-loop state instead of a stale pre-loop value.

The source profile accepts only single-name assignments that already satisfy
the restricted expression vocabulary and preserve the binding's IR type.
Scalar, index, and bool values are eligible. Tensor and tensor-view state,
quantum effects, loop-target shadowing, augmented assignment, and rebinding
from nested loops or conditional regions remain unsupported and fail closed.
Locals created only inside the loop remain region-local and do not escape.

Dynamic lowering requires no new Runtime representation: bounded loops are
still specialized and unrolled, each iteration receives the prior carried
tuple, and emitted RX/RY instructions continue to use ordered Core `Parameter`
slots. Measurement inside a loop and all finite-shot gradients remain outside
the profile. Runtime and Simulation receive the same bound `CircuitIR` as in
Phase 8 and therefore require no source changes.

Phase 9 acceptance is complete when verified IR exposes matching carried
signatures, accumulated scalar angles and index-selected wires survive
unrolling, post-loop operations consume explicit loop results, and tensor
carry or loop-target shadowing is rejected. This phase remains private and
makes no public API, default-path, general Python, accelerator, distributed,
capacity, or performance claim.

## Phase 10 explicit branch-carried classical-state authorization

Phase 10 may merge direct rebinding of existing scalar, index, and bool locals
across an `if/else`. Capture computes one deterministic carried-name list and
uses it for the `scf.if` operands and results and for both region signatures.
Each side yields its selected values plus the linear quantum effect. When only
one side assigns a name, the other side explicitly yields the unchanged region
argument, preserving Python's existing-binding semantics without a hidden
mutable cell.

The merged `scf.if` results replace the outer bindings, so subsequent gate
parameters, wire expressions, and compile-time-resolved conditions consume
SSA results rather than pre-branch values. Assignments must be direct branch
body single-name assignments, preserve the IR type, and remain within the
restricted expression vocabulary. Tensor and tensor-view state, augmented
assignment, and writes performed by nested branch or loop regions fail closed.

For dynamic-session lowering, branch-carried classical state is accepted only
when the predicate is resolved during specialization. A mid-circuit
measurement predicate still lowers both quantum-effect-only branches into
classical conditions; exporting different scalar, index, or bool values from
those branches would require a Runtime classical-expression representation
that Core `CircuitIR` does not currently own. Compiler therefore rejects that
case with a dedicated diagnostic before branch execution.

Phase 10 acceptance is complete when both specialized paths carry scalar,
index, and bool values into post-branch gates and measurement wires, one-sided
assignment has explicit pass-through, and tensor, nested, or
measurement-dependent carry fails closed. Runtime and Simulation remain
unchanged. The feature stays private and makes no public API, default-path,
general Python, finite-shot-gradient, accelerator, distributed, capacity, or
performance claim.

## Phase 11 nested structured-state propagation authorization

Phase 11 may discover writes to an outer scalar, index, or bool recursively
through supported nested `if` and `for` statements. A name updated in an inner
region is included in the carried signature of every enclosing structured
operation. Each level therefore receives and yields an explicit SSA value;
there is no closure mutation, runtime cell, global table, or stale outer-value
fallback.

The first nested profiles are an outer bounded loop containing an inner branch
and an outer compile-time-resolved branch containing an inner bounded loop.
Within those profiles, scalar angles, index wires, and bool selectors may flow
from the entry block through nested region arguments and results into later
gates and measurements. Region-local names remain local unless they correspond
to a binding visible at the enclosing operation.

Lowering continues to specialize predicates and bounded trip counts. Nested
loop iterations contribute to the same cumulative unroll ceiling; no nested
path bypasses the configured bound. Quantum effects remain linear at every
region boundary. A measurement-dependent branch still cannot export classical
state, and measurement inside any loop remains unsupported. Tensor carried
state and loop-target shadowing also continue to fail closed.

Phase 11 acceptance is complete when loop-around-branch and branch-around-loop
programs produce matching nested signatures, deterministic gate traces, and
expected trajectory outcomes. Runtime and Simulation remain unchanged. The
feature stays private and makes no public API, default-path, general Python,
finite-shot-gradient, accelerator, distributed, capacity, or performance
claim.

## Phase 12 measurement-boolean predicate authorization

Phase 12 may add `arith.not` and `arith.and` to the private Program IR and may
apply existing `arith.cmp` to equal bool values. Source capture accepts `not`
on a bool, an `and` of two or more bool expressions, and `==` or `!=` between a
measurement-derived bool and a bool constant. Measurements used in a composed
expression must first be assigned to local names; inline measurement calls are
rejected so capture never changes Python short-circuit measurement semantics.

Dynamic lowering retains measurement values symbolically as classical-bit
literals. Negation flips one literal's expected value, bool comparison either
preserves or flips it, and conjunction merges distinct literals into one
sorted tuple. Contradictory literals specialize to false. The resulting tuple
is stored in existing Core instruction `conditions` metadata, which both local
trajectory strategies already evaluate as a conjunction. This phase therefore
adds no Runtime or Simulation representation.

The complement of a multi-literal conjunction is a disjunction and cannot be
represented by one existing conditions tuple. Such a conjunction may guard
quantum work only in its true branch; a non-empty quantum else branch fails
closed. Boolean `or`, negated conjunction, measurement-to-measurement
comparison, repeated testing under an enclosing condition, and classical
values escaping a measurement-dependent branch remain unsupported.

Phase 12 acceptance is complete when `first and not second` controls a gate
exactly for classical register values `(1, 0)`, a bool comparison produces
complementary true/false gate conditions, and reference and batched execution
agree with every shot's recorded measurement values. The feature stays private
and makes no public API, default-path, finite-shot-gradient, accelerator,
distributed, capacity, or performance claim.

## Phase 13 bounded general measurement-predicate authorization

Phase 13 may add `arith.or` and extend symbolic measurement predicates to a
canonical disjunctive normal form (DNF): an ordered disjunction of ordered
conjunctions of classical-bit literals. Capture accepts `and`, `or`, `not`,
bool equality/inequality, and equality/inequality between two previously
assigned measurement booleans. Inline measurement composition remains
rejected so Python short-circuit evaluation cannot silently change which
measurements occur.

Dynamic lowering computes exact conjunction, disjunction, complement, XOR,
and XNOR over the canonical DNF. It removes contradictory, duplicate, and
subsumed clauses and sorts all remaining literals and clauses. Expansion is
bounded by `max_condition_clauses`, defaulting to 64; exceeding the bound fails
before Runtime execution. A single conjunction retains the existing
`conditions` metadata. Two or more clauses use private `condition_clauses`
metadata, evaluated as OR-of-AND by both local trajectory strategies.

This representation permits exact quantum work in both sides of a general
measurement-dependent `if`, including nested retests that simplify under an
enclosing predicate. Conditional measurement and measurement-dependent
classical values escaping a branch remain unsupported. Provider dialects that
do not explicitly understand DNF reject complex predicates instead of
exporting them as unconditional gates.

Phase 13 acceptance requires shot-wise agreement for disjunction, negated
conjunction, XOR/XNOR, and complementary quantum branches; deterministic
canonical metadata; explicit rejection of malformed or excessive clause sets;
and no regression to simple conjunctions. The feature stays private and makes
no public API, default-path, finite-shot-gradient, accelerator, distributed,
capacity, or performance claim.

## Phase 14 measurement-dependent SSA merge authorization

Phase 14 may preserve scalar, index, and bool results from a
measurement-dependent `scf.if` as bounded condition-partitioned SSA cases.
Each case pairs one runtime value with the canonical measurement predicate
under which that value exists. Nested conditional values are flattened, dead
cases are removed, and identical primitive or shared runtime values may be
coalesced by disjoining their predicates.

Arithmetic, remainder, Boolean composition, and comparisons distribute over
the bounded case product. A later quantum gate with a conditional parameter or
wire is split into mutually exclusive Core instructions carrying the matching
classical conditions. Subsequent structured branches and statically bounded
loops may consume and carry these values. This preserves per-shot value flow
without adding a mutable classical store or a second public program IR.

The number of live value cases shares the configured
`max_condition_clauses` ceiling and fails closed when exceeded. Conditional
measurement, measurement-dependent loop bounds, measurement-dependent program
returns, and finite-shot gradients remain unsupported. Phase 14 changes no
public API, default path, or Core serialization schema and makes no performance
claim.

## Phase 15 fixed-round syndrome-feedback authorization

Phase 15 may execute `quantum.measure` inside a statically bounded `scf.for`
and may add `quantum.reset` as an index-and-effect Program IR operation.
Dynamic lowering fully unrolls the bounded loop, assigns every measurement a
distinct dense classical bit in program order, and carries the latest syndrome
SSA value across iterations and out of the loop.

Each round may measure an ancilla, evaluate the resulting Boolean predicate,
apply immediate conditioned correction gates, and unconditionally reset the
ancilla before the next round. Runtime retains ownership of collapse, reset,
the per-shot classical register, and conditional gate execution. Compiler
does not sample syndrome values while lowering.

The profile is bounded independently by `max_unrolled_iterations`,
`max_dynamic_measurements`, and `max_condition_clauses`, with defaults of
10,000, 4,096, and 64. Exceeding any configured bound fails before Runtime.
Conditional measurement or reset, measurement-dependent loop termination,
finite-shot gradients, and provider execution remain unsupported.

Phase 15 acceptance requires a fixed three-round correction program in which
the first syndrome detects an injected data error, its conditioned correction
clears the error, later syndromes remain clear, every ancilla reset succeeds,
and reference and batched trajectories agree. This is a bounded QEC control
slice, not a logical-error-rate, threshold, decoder, hardware-latency, or
fault-tolerance claim.

## Phase 16 QEC-domain repetition-memory authorization

Phase 16 may add an experimental `flagquantum.qec` domain for QEC-owned code
profiles, syndrome and detection-event records, decoder contracts, correction
decisions, workflows, and logical-result analysis. Generic circuit, compiler,
runtime, simulation, noise, and provider semantics remain with their existing
owners. No separate FTOC domain or duplicate execution authority is created.

The first workflow is restricted to three data qubits, two adjacent Z-parity
checks, reusable ancillas, a fixed positive round count, and zero or one
deterministically injected X error. It composes Phase 15 lowering with the local
dynamic session. The compiled program executes a fixed lookup correction;
typed `Decoder` replacement is presently limited to post-execution analysis of
recorded syndromes and must not be described as a real-time callback.

Phase 16 acceptance requires correction of each single-data-qubit injection,
zero-error preservation, consistent reference and batched trajectories,
temporal detection events, typed shot and result records, replaceable analysis
decoder conformance, and fail-closed validation. The subpackage is not exported
from the stable root. No general-code, realistic-noise, logical-suppression,
threshold, controller-latency, provider, gradient, distributed, capacity,
performance, or fault-tolerance claim is authorized.

## Phase 17 timed-error and offline-frame authorization

Phase 17 may replace the initial pre-run error selector with a canonical QEC
`ErrorSchedule`. A bounded schedule identifies deterministic X errors by data
wire and fixed round. Lowering specializes those validated events into the
unrolled program at the start of each round, before parity checks. Duplicate
round/wire events, events outside the run, non-X errors, and schedules above 64
events fail before Runtime execution.

The repetition workflow exposes two separate policies. `compiled_lookup`
retains immediate measurement-conditioned X feedback. `offline_pauli_frame`
executes no physical feedback and applies the decoder's readout frame only
after execution. The decoder receives the complete, ordered syndrome history
and returns typed corrections plus a parity-reduced Pauli frame. Each shot
records actual compiled feedback separately from decoder advice.

Phase 17 acceptance requires every single data-wire error in any of three
rounds to be corrected by both policies, exact syndrome/detection-event
histories, and explicit logical failure for two same-round errors. This does
not authorize stochastic or measurement noise, a Runtime decoder callback,
logical-error suppression, threshold, general stabilizer-code, hardware,
gradient, distributed, capacity, performance, or fault-tolerance claims.

## Phase 18 bounded dynamic-noise authorization

Phase 18 may pass the existing backend-neutral `NoiseModel` into local dynamic
execution. Runtime owns seeded random streams, placement after matching
actually executed gates, true-versus-observed measurement flow, conditional
feedback from observed bits, and event accounting. Simulation owns the
numerical bit-flip and readout-sampling kernels. No second noise model or
compiler-owned stochastic executor is authorized.

The accepted channel profile is one-wire independent bit flips. Independent
readout confusion may affect explicit measurements and final sampling; state
collapse follows the true result while the classical register records the
observed result. Reference and batched trajectory strategies must preserve
these semantics. General Kraus channels, correlated readout, device-profile
timing noise, noise on reset, and noisy gradients fail closed.

QEC may map this generic capability onto parity-check circuit locations and
emit finite-shot logical-error observations. The mapping must disclose that
the middle repetition-code data wire has two CNOT noise opportunities per
round while edge wires have one. Phase 18 does not authorize calibrated-device,
logical-suppression, threshold, real-time-decoder, provider, scalability,
performance, or fault-tolerance claims.

## Phase 19 bounded Runtime decoder-feedback authorization

Phase 19 may add private, framework-neutral measurement decision points to the
local dynamic trajectory executor. Runtime owns point validation, true and
observed measurement records, invocation order, bounded physical-X or
Pauli-frame-X actions, frame evolution, and shot-resolved traces. Runtime must
not import QEC concepts. Batched feedback fails closed; `auto` selects the
trajectory path when a private feedback plan is present.

QEC may adapt those records to a `StreamingDecoder` that consumes all syndrome
rounds available at the current decision. The repetition reference may execute
immediate physical correction or update an X Pauli frame. A frame adjusts the
interpretation of later parity checks and final data readout; it does not alter
the quantum state. Replacing the streaming decoder must observably change
continued execution, rather than merely changing post-run analysis.

Acceptance requires agreement of logical outcomes across compiled lookup,
Runtime physical feedback, and Runtime frame feedback for bounded single-error
cases; shot traces separating true from observed bits; and rejection of
unmeasured decision bits, out-of-range actions, and batched feedback. This is a
local synchronous reference loop, not a stable plugin type or hard-real-time
controller contract. General codes, provider execution, latency, gradients,
logical suppression, thresholds, scale, performance, and fault tolerance
remain outside the authorized claim.

## Phase 20 bounded temporal-decoder authorization

Phase 20 may add a repetition-code `RepetitionTemporalDecoder` that validates
and consumes all syndrome and detection-event records available at each Runtime
decision. The accepted reference rule requires the same non-zero two-check
syndrome in two consecutive rounds, with no new detection event at the second
boundary, before issuing a correction. A one-round readout excursion followed
by the matching return detection event produces no correction.

Both physical-X and Pauli-frame-X Runtime policies may use this decoder. Data
errors introduced no later than the penultimate round have a two-round
confirmation window. An error first observed in the terminal round remains
visible and explicitly unconfirmed. Inconsistent detection-event histories
fail before a correction is returned.

Acceptance requires exact persistent-data and isolated-readout oracles,
physical and frame correction of confirmable single-data errors, explicit
terminal-round non-correction, and a seeded finite-shot observation that the
temporal rule emits fewer spurious actions than immediate lookup under the
checked readout-noise profile. This does not authorize a maximum-likelihood
decoder, arbitrary measurement-error tolerance, logical suppression, a
threshold, general codes, provider/hard-real-time execution, gradients,
scalability, performance, or fault tolerance.

## Phase 21 verified Program IR normalization authorization

Phase 21 may add a bounded private pass pipeline between `HybridProgram`
verification and the existing static or dynamic lowering paths. The pipeline
keeps `HybridProgram` as the authoritative program representation. It may
compute constant-value, whole-program SSA-use, and operation-count facts; fold
supported constant-only arithmetic; and remove unused `arith.constant`
operations after folding.

Verification runs before the pipeline and after every transformation. A pass
must yield a `HybridProgram`, directly or through a verified private outcome,
and a pipeline longer than 32 passes fails before transformation. Lowered
results retain the source identity, optimized identity,
pass names, and before/after operation counts. A private unoptimized path is
kept solely as a differential correctness oracle.

Operations with possible runtime failure are not removed simply because their
results are unused. The phase does not authorize a second target IR, target
dialect, general optimizer or plugin registry, stable pass extension API,
public export, device-specific optimization, or performance claim. Acceptance
requires deterministic and idempotent normalization, static and dynamic
lowering equivalence against the unoptimized oracle, preserved parameter
autograd edges, and fail-closed invalid-pass boundaries.

## Phase 22 structured-control-flow simplification authorization

Phase 22 may simplify only structured control whose outcome is established by
the verified constant analysis. A constant `scf.if` may be replaced by the
selected region body. Its explicit classical carried values and linear quantum
effect are rewired from the operation results to the selected `scf.yield`
operands. A statically empty `scf.for` may be removed by rewiring its results to
the initial carried operands.

The transformation must preserve value types, definition-before-use, and
single-use linear effects and must pass the ordinary verifier before later
lowering. Nonconstant branches and loops that may execute remain represented;
measurement-dependent control must not be resolved by Compiler. Static and
dynamic optimized lowering must agree with their unoptimized differential
oracles.

This phase does not authorize general loop unrolling, loop-invariant code
motion, speculative execution, target-specific transformation, a target IR,
stable pass plugins, public exports, or performance claims.

## Phase 23 bounded constant-loop unroll authorization

Phase 23 may unroll a direct entry-block `scf.for` only when constant analysis
establishes integer lower, upper, and nonzero step values. The default pass may
expand at most eight iterations per loop and at most 256 operations across the
pipeline. A loop outside either budget remains structured for the existing
specialization or dynamic-lowering path.

Each expanded iteration has fresh SSA definitions and an explicit induction
constant. Classical carried values and the linear quantum effect are chained
from one cloned yield to the next. The final values replace the original loop
results. The transformed program must pass the ordinary verifier and repeated
normalization must be an identity transformation.

The number of compile-time-expanded iterations is retained in pass evidence
and initialized into the existing lowering counter. Therefore optimization may
not bypass a caller's `max_unrolled_iterations` policy. Acceptance requires
optimized/unoptimized `CircuitIR` agreement, parameter-value agreement,
preserved autograd edges, and matching limit failures. Nested-loop expansion,
general unrolling, target-specific scheduling, public APIs, and performance
claims remain unauthorized.

## Phase 24 pass-audit and differential-verification authorization

Phase 24 may split the private optimizer implementation into four local
responsibilities: verified analysis, SSA rewrite helpers, concrete
transformations, and pipeline/audit orchestration. This is an internal
maintainability change and must not create another IR, pipeline authority, or
public compiler surface.

Concrete passes may return a private immutable outcome containing the verified
program, non-negative integer statistics, and deterministic nonempty remarks.
Records expose actual fold, removal, branch, loop, expansion, and budget-skip
behavior. Audit values do not affect program semantic identity. A direct
`HybridProgram` result remains accepted for narrow internal test passes.

Acceptance requires fixed-seed differential programs covering constant
branches, bounded loops, carried scalar/index values, ordered quantum effects,
parameter binding, statevector execution, and adjoint VJP. Optimized and
unoptimized paths must agree, and a second optimization run must be a fixed
point. Negative runtime loop steps remain unsupported and must fail identically
with optimization enabled or disabled. No new optimization semantics, target
IR, public API, or performance claim is authorized.

## Phase 25 capability-driven target-legality authorization

Phase 25 may add a private Compiler-owned legality stage after hybrid lowering
has produced the existing Core-owned `CircuitIR`. The stage derives mandatory
logical-qubit capacity, maximum program-operation count, effective precision,
measurement-result profile, and finite-shot capacity requirements solely from
that circuit. It validates every distinct circuit opcode through the existing
operator-lowering registry and matches the derived `RequirementSet` against one
explicit Core-owned `TargetCapabilitySnapshot`.

Success retains the exact input `CircuitIR` and records the backend, requirement
identity, snapshot identity, resolved lowering strategies, pure match result,
and a deterministic legalization identity. Failure in operator lowering or any
mandatory target capability fails before emission. All fallback authorization
axes remain false. The stage does not select a backend or target, probe a
device, mutate a circuit, execute numerical work, or move Runtime planning into
Compiler.

The current Core `gates.native` vocabulary admits richer provider descriptors
whose parameter-domain compatibility is not expressible by its conservative
collection matcher. Phase 25 therefore uses the existing Compiler-owned
operator-lowering registry for opcode legality and does not claim native-gate
legalization. Native gate-set normalization, decomposition, topology routing,
scheduling, artifact-profile negotiation, and target emission remain later,
separately verified stages. No `TargetIR`, public export, stable API change,
automatic default-path integration, or performance claim is authorized.

## Phase 26 evidenced native-gate legalization authorization

Phase 26 may normalize the target snapshot's verified `gates.native` fact into
Compiler-owned opcode and parameter-name descriptors. Opcode strings inherit
the canonical parameter names in the existing Core operator schema; descriptor
objects must explicitly list supported parameter names. Multiple descriptors
for one opcode remain separate variants and must not be unioned into capability
that no target variant actually declares.

An already native instruction remains the same object. The initial verified
decomposition set is deliberately closed: X to H-Z-H, RX to H-RZ-H, and RY to
Sdg-H-RZ-H-S. Replacements reuse the exact parameter objects and copy dynamic
condition metadata to every generated instruction. The transformed artifact is
still the existing Core-owned `CircuitIR`. At most 256 additional instructions
may be introduced by default.

Native-gate evidence validity is checked before transformation. Every generated
instruction must match one complete native descriptor variant. Unsupported
source gates, missing basis gates, incomplete parameter descriptors, custom
matrices, stale evidence, and expansion overflow fail closed. Phase 25 then
derives operation-count and other target requirements from the legalized
circuit, so decomposition cannot bypass target limits.

This phase does not claim parameter ranges or periodicity, arbitrary synthesis,
approximation, calibrated fidelity, coupling-map legality, routing, scheduling,
timing, pulse generation, target-format emission, or provider execution. It
adds no TargetIR, public export, stable API change, default-path change, or
performance claim.

## Phase 27 bounded topology-legalization authorization

Phase 27 may route a Core-owned `CircuitIR` against one explicit Compiler
`CouplingMap` using the existing restore-after-each-gate or persistent-layout
strategies. `auto` may select between those two strategies using the existing
deterministic cost estimate. The legality stage clones the coupling map before
routing so mutable shortest-path cache history cannot affect the artifact or
its audit identity.

Every two-wire instruction in the routed result must occupy a coupling edge.
Nonlocal channels fail closed because the existing router deliberately skips
channel insertion. The final logical-to-physical layout must be identity and
the routing metadata must report successful restoration and a non-negative
inserted-SWAP count. Routing may add at most 256 instructions by default.
Circuit parameters and their autograd references must survive remapping.

Target legalization orders this stage before native-gate legalization. Routing
SWAPs may then use the exact SWAP-to-three-CX decomposition, after which native
gate and final resource checks observe the real expanded program size. The
topology audit binds source and routed circuit identities, target snapshot,
coupling topology, and selected strategy.

Target Capabilities v1 currently has no coupling-map fact. The explicit map is
therefore caller-supplied and merely bound to the selected snapshot identity;
Phase 27 does not claim that the snapshot proves its provenance or freshness.
Directed couplings, physical ancilla allocation, placement, calibration,
fidelity-aware routing, crosstalk, duration, parallel schedule, and provider
execution remain outside the contract. No TargetIR, public API, default-path
change, or performance claim is authorized.

## Phase 28 dependency-preserving logical-scheduling authorization

Phase 28 may construct immutable scheduling evidence over the final legalized
Core-owned `CircuitIR`. It uses deterministic ASAP placement with per-wire
source-order dependencies and explicit classical producer-to-consumer edges.
The evidence records instruction indices rather than copying instructions into
a second executable representation, and it binds the final circuit content and
selected target-snapshot identity.

Dynamic measurement and reset operations, conditioned operations, and channel
instructions are conservative global barriers. Classical conditions must be
well-formed binary bit/value pairs, and every referenced bit must have a prior
measurement producer. Empty programs have depth zero; nonempty layers contain
no repeated wire. An optional non-negative maximum logical depth fails closed
after routing and native decomposition have established the actual final
program.

Depth is a logical unit-layer count only. This phase does not infer gate
duration, measurement or feedback latency, concurrent hardware support,
crosstalk, fidelity, calibration, pulse timing, or makespan. It does not emit a
target program, execute the schedule, create TargetIR, add public exports,
change the default path, or make a performance claim.

## Phase 29 verified deterministic target-text-emission authorization

Phase 29 may invoke the existing OpenQASM 2, OpenQASM 3, or QCIS text emitter
only after receiving a successful `TargetLegalizationResult`. A closed profile
table binds each exact format version and media type to its required backend.
The immutable result records emitted text, its SHA-256 digest, final circuit
content, selected target snapshot, target legalization, logical schedule, and a
deterministic emission identity.

The scheduler is rerun before emission and its identity must match the evidence
stored during legalization. The current lossless profile requires exactly one
terminal full-register samples request and rejects observable expectations,
dynamic measurement or reset, classical conditions, channels, arbitrary
matrices, unbound parameters, unknown profiles, and backend/profile mismatch.
Finite shots remain execution-request information and are not encoded into the
program text.

`TargetEmissionResult` is an in-process Compiler result, not a serializable
artifact envelope or trust boundary. This phase neither changes nor repurposes
Core `ProgramArtifact` v1, and it does not restore the removed historical
`SealedExecutableArtifact`. It adds no target submission, provider credentials,
Runtime adapter, deployment package, TargetIR, public export, default-path
change, or performance claim.

## Phase 30 strict target-text-conformance authorization

Phase 30 may verify a Phase 29 emission against its exact target-legalization
input and independently parse the emitted static subset. Exact reproduction
must match the complete immutable emission result before parsing. OpenQASM 2
and 3 parsing validates canonical headers, declarations, operations,
parameters, wire ranges, and terminal full-register measurement. QCIS parsing
accepts only the native instruction forms produced by the existing emitter and
maps them into equivalent Core operations.

The reconstructed program is the existing Core-owned `CircuitIR`. Its terminal
samples request has unspecified shots because target text does not encode shot
count. The conformance identity binds the emission identity, exact format,
parser version, and reconstructed circuit content. Payload or identity
tampering, a different legalization input, unknown syntax, invalid parameters,
or out-of-range wires fail closed.

Compiler performs no numerical execution. Bounded statevector comparisons are
test evidence executed through the Simulation-owned implementation and do not
constitute target or hardware certification. This phase introduces no second
IR, artifact envelope, Runtime adapter, deployment path, provider submission,
TargetIR, public export, default-path change, or performance claim.

## Phase 31 ProgramArtifact-v2 proposal authorization

Phase 31 may produce API Change Proposal 022, a machine-readable candidate
contract, and compatibility tests for an executable-text profile in the
existing `flagquantum.program_artifact` lineage. It freezes one v1 fixture and
its current content hash without modifying the v1 implementation.

The proposal may specify closed OpenQASM 2, OpenQASM 3, and QCIS 1 profiles;
UTF-8 payload limits and a declared payload digest; final-circuit, target,
legalization, schedule, emission, conformance, and envelope identities;
structured target requirements; fully bound parameters; terminal sample result
shape; strict canonical JSON; security exclusions; ownership; and explicit
version dispatch. Shots and provider lifecycle data remain separate.

The proposal is not approval. Until the exact approval token is supplied, no
v2 reader/writer, Core schema change, Compiler adapter, Runtime adapter,
Deployment integration, public export, or provider submission is authorized.
The private Phase 29/30 records remain the only implemented target-text path.

## Phase 31.1 circuit-profile proposal revision authorization

Phase 31.1 may revise Proposal 022 before approval to add a strict
`circuit-ir-1.0` profile alongside the three executable-text profiles. The
circuit payload must round-trip through Core `CircuitIR`; its canonical payload
digest and CircuitIR content hash have separate roles but equal values. Circuit
artifacts are target-independent, so requirements, target, compilation, and
result-schema fields are null.

The revision may specify that all new artifact writes use v2 after approval and
v1 becomes read-only. Automatic v1 migration is allowed only for circuit kind,
empty metadata, empty flat capability hints, empty opaque parent hashes, and a
strictly valid CircuitIR payload. All other v1 artifacts remain readable but
fail automatic migration. A pinned v2 circuit candidate records the proposed
canonical identities.

This revision was approved by the API owner on 2026-09-10 using the exact
Proposal 022 approval token.

## Phase 32 ProgramArtifact-v2 Core implementation authorization

Phase 32 implements only the approved Core-owned portion of Proposal 022:
the strict v2 circuit and executable profile model, canonical identities,
closed limits, explicit version dispatch, duplicate-key-safe JSON reading, and
the narrow v1 circuit migrator. The v1 class, reader behavior, serialization,
and content hash remain unchanged.

This phase does not add a Compiler construction adapter, Runtime compatibility
adapter, Deployment/provider conversion, public export, default-path change,
or performance claim. Those remain separate follow-on phases and cannot infer
authority merely from Core accepting the v2 envelope.

## Phase 33 Artifact-v2 vertical slice and local circuit execution

The approved v2 contract is connected through four bounded internal seams:
Compiler builds executable artifacts only from successful target conformance;
Runtime checks executable artifacts against the exact target snapshot and
request shots; Deployment prepares a side-effect-free provider handoff; and
Runtime executes fully bound `circuit-ir-1.0` artifacts through its existing
local planner and result contract.

Local artifact execution reconstructs only the canonical Core CircuitIR
payload. It records the artifact, payload, and circuit identities in result
provenance. Execution-request fields do not alter artifact identity. Symbolic
circuit artifacts fail until a separate binding contract exists. OpenQASM and
QCIS executable artifacts remain target-bound and must not be reinterpreted as
local simulator input.

This phase adds no provider submission, credential access, public export,
default-path change, new numerical kernel, or performance claim.

## Phase 34 Explicit symbolic circuit-artifact binding

Core may bind a symbolic `circuit-ir-1.0` artifact only from an exact mapping of
its declared parameter names to finite Python real scalars. Binding produces a
new fully bound v2 artifact and immutable in-process lineage evidence containing
the source identity, bound identity, sorted parameter names, and deterministic
binding identity. It does not mutate the source or extend the v2 envelope.

Missing or extra names, non-finite values, booleans, tensors, arrays, and
already-bound or non-circuit artifacts fail closed. In particular, tensors with
autograd state are never detached and serialized by this path. Differentiable
training remains on the existing ephemeral binding-table route. Runtime may
accept the binding result for local execution and records both source and
binding identities in result provenance.

This phase adds no executable-text parameter profile, provider submission,
public export, default-path change, or performance claim.

## Phase 35 Verified artifact-to-artifact target compilation

Compiler accepts either a fully bound `circuit-ir-1.0` artifact or the verified
binding result from Phase 34. A single internal entrypoint verifies the input
identity and then runs target capability legalization, optional topology
routing, native-gate legalization, logical scheduling, deterministic target
emission, strict conformance, and executable-artifact construction in that
order. No caller-facing flag can skip a stage.

The immutable result retains the actual input artifact object plus source,
binding, circuit-artifact, target-stage, executable-artifact, and complete
compilation identities. Its constructor verifies that the topology/native gate
chain starts from the input circuit and that every downstream identity agrees
with the final artifact.

This phase performs no numerical execution, target selection, provider
submission, credential access, public export, default-path change, or
performance claim.

## Phase 36 Physical mapping and dependency-schedule plan

Compiler composes the existing topology routing, native-gate legalization, and
dependency schedule into an immutable `PhysicalCircuitPlan`. The plan is a
verified view over the final Core-owned `CircuitIR`, not a new TargetIR or a
second instruction authority. Topology and native legalization results retain
their actual source `CircuitIR` objects so every final physical instruction can
be traced through its routed instruction to its logical source instruction.

Routing SWAPs carry exact source indexes and produce replayable layout
transitions. Every final two-wire instruction is checked against the supplied
coupling map. Final instructions record their source and topology indexes,
native replacement ordinal, physical wires, logical-source wires, dependency
layer, predecessors, and dependency kinds. The plan also records deterministic
depth, maximum parallel width, and one deterministic dependency critical path.

These layers are unit-time causal layers only. They make no claim about gate
duration, pulse timing, crosstalk, calibration quality, directed couplings,
physical ancilla allocation, or hardware performance. Artifact-to-artifact
compilation now requires the plan and includes its identity in the enclosing
compilation identity, while the approved ProgramArtifact v2 envelope remains
unchanged.

## Phase 37 Compilation evidence bundle handoff

Phase 37 defines a separate Core-owned, versioned compilation-evidence bundle
instead of modifying ProgramArtifact v2 or embedding compiler provenance in
free-form metadata. The bundle would serialize Phase 36 mapping, native-gate,
and dependency evidence and binds it to the actual source artifact, target
snapshot, executable artifact, and artifact-compilation identity.

The exact candidate schema is recorded in
`contracts/compilation-evidence-bundle-v1-candidate.json` and the decision is
specified by `API_CHANGE_PROPOSAL_023_COMPILATION_EVIDENCE_BUNDLE.md`. The
proposal deliberately excludes circuit payload duplication, TargetIR,
execution requests, provider state, timing and pulse claims, calibration-aware
optimization, and fault-tolerant expansion.

The API owner supplied the exact Proposal 023 token on 2026-09-10. Core now
implements the strict value model, canonical JSON, limits, nested validation,
and deterministic bundle identity. Compiler constructs and verifies the bundle
from the actual retained compilation objects and target snapshot. Runtime and
Deployment verify the decoded bundle against the actual fully bound source or
binding result, executable artifact, and target snapshot. Deployment dry-run
may carry the verified bundle beside the executable artifact, but it performs
no provider call and stores no credentials or task state. No public export or
default-path changes are included.

## Phase 38 Directional topology and explicit layout proposal

Phase 38 proposes migration of the proven historical Batch C directed-routing
semantics into the current single-`CircuitIR` compilation chain. The existing
Compiler `CouplingMap` remains undirected. A distinct, initially private
directed topology type would support complete-permutation initial placement,
weak-connectivity routing, direction-correct reverse-CX synthesis, and mandatory
identity final-layout restoration.

Because compilation-evidence 1.0 explicitly means undirected topology and
identity initial layout, the new semantics require explicit evidence version
2.0 rather than reinterpretation of existing fields. ProgramArtifact v1/v2,
public exports, and default execution remain unchanged. The candidate excludes
physical ancillas, partial layouts, observable remapping, calibration-aware
routing, timing/pulse claims, and fault-tolerant expansion.

The exact candidate is recorded in
`contracts/directional-topology-layout-v2-candidate.json` and the decision is
specified by
`docs/development/API_CHANGE_PROPOSAL_024_DIRECTIONAL_TOPOLOGY_LAYOUT.md`.
The API owner supplied the exact Proposal 024 token on 2026-09-10. The first
implementation slice now provides the private directed topology, explicit
placement and restoration, reverse-CX legalization, final native/directional
verification, physical-plan 2.0 lineage, and executable-artifact compilation.
Compilation-evidence 1.0 explicitly rejects these directed plans until the
separately tested Core 2.0 and Runtime/Deployment handoff is complete.

Phase 40 now implements the Core-owned strict version 2.0 values, canonical
encoding, explicit 1.0/2.0 reader dispatch, directed topology and rewrite-group
validation, and Compiler construction/verification from actual retained
objects. Version 1.0 remains unchanged. Runtime and Deployment version-2
handoff support remains a separate final slice.

Phase 41 completes that slice. Runtime validates either evidence version
against the actual fully bound source, target snapshot, and executable artifact.
Deployment may carry the same immutable bundle through its side-effect-free
dry run after Runtime verification. Neither layer rewrites evidence or performs
provider submission, and ProgramArtifact v1/v2 remain unchanged.

## Phase 42 Physical resource allocation proposal

Phase 42 proposes the next bounded physical-compilation capability: a source
program may use fewer logical wires than the target snapshot provides physical
slots, and deterministic routing may temporarily move logical state through an
initially idle slot. The additional slot is compiler routing workspace in the
standard zero state, not a user-visible logical wire, QEC syndrome ancilla, or
fault-tolerant resource claim.

The current version-2 contracts cannot express this without ambiguity. Their
layout is a complete permutation at equal capacity, while ProgramArtifact v2
requires dense full-register samples. Merely increasing final `CircuitIR.n_wires`
would therefore expose physical workspace as user output. The proposed version-3
profile separates logical wires, dense compilation-local physical slots,
adapter-owned provider identifiers, and ordered result positions. It adds nullable
physical occupancy, an injective sparse placement, mandatory workspace cleanup,
and an authenticated projection from physical result slots into logical-wire order.

The proposal preserves ProgramArtifact v1/v2 and compilation evidence 1.0/2.0.
It introduces no second instruction authority and excludes provider identifier
binding, provider submission, mid-circuit ancilla reuse, arbitrary ancilla states,
observable remapping, calibration-aware allocation, and all QEC/FTOC claims. The
exact candidate is recorded in
`contracts/physical-resource-allocation-v3-candidate.json`; implementation requires
the exact approval token specified by
`docs/development/API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION.md`.

The API owner supplied that exact token on 2026-09-10. Phase 43 implements the
first Compiler-only slice for directed targets. An injective logical placement may
leave physical slots idle; every routing SWAP records both the logical projection
and complete nullable physical occupancy before and after the transition. Cleanup
applies the inverse routing-SWAP sequence, restores the declared result placement,
and proves that every non-result slot is idle again. The routed Core `CircuitIR`
has physical width, while terminal sample wires name only the ordered logical
result slots.

The equal-capacity version-2 route is unchanged. Allocated programs remain blocked
from physical-plan construction with an explicit version-3 requirement, so they
cannot accidentally enter ProgramArtifact v2 or compilation-evidence 2.0. The
slice supports only static unitary instructions and full logical-register samples;
observables, dynamic operations, channels, provider identifiers, and QEC/FTOC
ancilla semantics fail closed.

Phase 44 promotes those retained Compiler facts into private
`PhysicalCircuitPlan` version 3.0. Each mapping transition now carries and replays
both the logical-to-physical projection and the complete nullable
physical-to-logical occupancy. The plan validates logical and physical counts,
coupling size, injection uniqueness, transition continuity, pre-restore state,
final result placement, null workspace restoration, instruction lineage,
direction legality, scheduling, and the allocation identity retained by topology
legalization.

The version-3 plan identity covers all allocation fields and per-transition
occupancy. Equal-capacity undirected and directed plans still produce the exact
version-1 and version-2 identity payloads respectively; they reject version-3
allocation fields. ProgramArtifact 3.0 and compilation-evidence 3.0 remain
unimplemented, so no allocated executable can yet pass the target-emission and
artifact handoff boundary.

Phase 45 implements that executable boundary as Core `ProgramArtifact` version
3.0. The closed artifact remains fully bound and content addressed, requires a
target snapshot, and additionally binds the actual physical-plan and allocation
identities. Its result schema records dense logical wires, unique physical result
slots, logical-wire ordering, and execution-request shot ownership. Version
dispatch is explicit; ProgramArtifact v1/v2 constructors, bytes, readers, and
profiles are unchanged.

Compiler target emission now recognizes only the authenticated allocated-routing
profile. OpenQASM declares the full physical quantum register but a logical-width
classical register, then measures each declared physical result slot into its
logical result position. The strict conformance parser reproduces that projection
and rejects register-width, ordering, or terminal-measurement changes. QCIS fails
closed because its current text profile cannot encode the projection. Artifact
compilation returns ProgramArtifact v3 only when bound to an actual physical plan
3.0; equal-capacity plans continue returning v2. Compilation-evidence v3 and all
Runtime/Deployment consumption remain outside this phase.

Phase 46 implements Core compilation-evidence version 3.0 without changing the
version-1/2 models. Its strict physical-plan value separates logical capacity from
physical capacity, validates injective layouts and complete nullable occupancy,
replays both state representations across every SWAP, authenticates the first
pre-restore state, and requires the declared logical result slots to match the
final logical layout. The allocation identity is recomputed from the closed
standard-zero/inverse-SWAP profile, and the physical-plan identity covers every
new allocation and occupancy field.

Compiler bundle construction selects version 3.0 only from an actual allocated
`PhysicalCircuitPlan` 3.0 and carries the existing source, target, output,
direction-legality, native-lineage, and schedule evidence forward unchanged.
Compiler verification reconstructs the expected bundle from retained objects and
requires exact equality. JSON dispatch is explicit for versions 1.0, 2.0, and
3.0. Runtime and Deployment still reject version 3.0; provider submission,
provider-qubit binding, general ancillas, QEC, and fault-tolerant resource claims
remain outside the implemented boundary.

Phase 47 extends the Runtime read-only boundary to matching ProgramArtifact v3 and
compilation-evidence 3.0 values. Runtime preflight applies the same target snapshot,
artifact-profile, shot-limit, and capability checks used for v2. Handoff validation
requires v3 artifact/evidence version agreement and cross-checks physical-plan,
allocation, schedule, target, executable, logical-wire, physical-result-slot,
ordering, and shot-source identities or fields.

The private result validator accepts only a tensor that already has the declared
logical width in its last dimension and returns that same object. It does not use
the physical-slot list to gather, reorder, truncate, or otherwise repair provider
output: ordered OpenQASM emission owns physical-to-classical placement, and the
adapter must return those classical positions in logical-wire order. Scalar and
physical-width results fail closed. Deployment v3 handoff and provider execution
remain outside this phase.

Phase 48 completes the approved private version-3 profile at the Deployment dry-run
boundary. A ProgramArtifact v3 cannot enter that handoff without its source and a
matching compilation-evidence 3.0 bundle. Deployment delegates target, capability,
shot, lineage, plan, allocation, and result-projection validation to the Runtime
checks, then retains the exact immutable artifact and evidence objects.

The dry-run view exposes logical result width and compilation-local physical result
slots. Those slots are not provider qubit identifiers and are not translated or
enriched. The handoff contains no credentials or provider task identifier, performs
no network operation, and cannot claim target acceptance or execution. Public root
exports and default execution remain unchanged.

Phase 49 closes Proposal 025 with a machine-readable exit audit and completion
review. The audit binds the six implementation commits, approved acceptance
classes, stable-export result, unchanged default path, and explicitly excluded
production, provider-binding, submission, general-ancilla, and QEC/FTOC claims.
Automated checks inspect the actual stable `__all__` manifests rather than relying
only on documentation. No runtime behavior or public surface is added in this
phase; every expansion beyond the private technical profile requires a separate
proposal and approval.

## Phase 50 public artifact lifecycle proposal

Phase 50 records Proposal 026 and its exact machine-readable candidate without
implementing or exporting an API. The proposed first lifecycle is a read-only
preview at `flagquantum.experimental.artifacts`, expressed through role-named
frozen views and strict load/dump functions. Concrete v1/v2/v3 implementation
classes remain private so serialized schema revisions do not become the public
type hierarchy.

The proposed preview excludes constructors, binding, Compiler workflows, Runtime
verification, Deployment preparation, target-capability APIs, provider binding or
submission, and QEC/FTOC semantics. It requires the exact approval token in
`docs/development/API_CHANGE_PROPOSAL_026_ARTIFACT_PUBLIC_LIFECYCLE.md` before any
module or symbol is added.
