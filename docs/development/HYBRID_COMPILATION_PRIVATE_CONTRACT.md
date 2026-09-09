# Private hybrid compilation contract

Status: Phases 1-18 implemented and verified under the repository owner's
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
