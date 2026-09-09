# Python-first hybrid quantum-classical compilation execution plan

Status: **Phases 1-18 bounded vertical slices complete**
Owner: Compiler, with Integration approval for cross-domain contracts
Initial target: local CPU `single_device_fast_path`
Implementation language constraint: no FlagQuantum-authored C++ in Phases 0-6
External compiler constraint: Catalyst is out of scope for this plan

## 1. Outcome

Deliver a FlagQuantum-owned path that captures, validates, specializes, lowers,
executes, and differentiates a bounded hybrid quantum-classical Python program.
The first complete scenario is a parameterized cost function containing nested
loops and data-dependent branches that choose quantum gates, followed by an
expectation value and a first-order gradient.

The first accepted vertical slice must prove this flow:

```text
restricted Python function
  -> private hybrid Program IR
  -> verified structured control flow
  -> selected quantum operation trace
  -> existing CircuitIR
  -> existing compiler and local statevector simulation
  -> scalar expectation
  -> VJP back to the original tensor parameters
```

The work is not complete when new IR classes exist. It is complete only when the
scenario above runs end to end, fails closed outside its declared subset, and
passes independent forward and gradient differentials.

## 2. Product and architecture constraints

This plan obeys the following repository constraints:

- `CircuitIR` remains the stable public exchange representation and the source
  consumed by existing simulation paths.
- The hybrid IR is private until a separate public API proposal is approved.
- Existing `fq.run`, `fq.plan`, `fq.Module`, `fq.compile`, result types, and
  serialized schemas do not change in the initial phases.
- The existing local CPU and single-device paths must not incur hybrid compiler
  overhead unless the new path is explicitly selected.
- Compiler captures and transforms programs; Runtime owns execution lifecycle;
  Simulation owns numerical kernels; Ecosystem owns external-framework adapters.
- Unsupported control flow, mutation, operations, gradients, measurements,
  shapes, and effects fail before execution whenever they are statically known.
- No performance, scale, hardware, or production claim is made from CPU
  conformance tests.
- No duplicate plugin registry, backend manager, execution result, or public IR
  is introduced.

## 3. Explicit non-goals

The following are excluded from the first six phases:

- Catalyst integration or compatibility claims;
- a C++ compiler, C++ runtime, native device ABI, or Python-free executable;
- general Python compilation;
- arbitrary Python objects, reflection, generators, exceptions, I/O, networking,
  or mutation inside captured functions;
- provider submission, QPU credentials, queue management, or cloud execution;
- QIR, pulse, timing, or controller firmware generation;
- distributed scalability claims;
- finite-shot, noisy, or mid-circuit-measurement gradients;
- higher-order differentiation;
- automatic public API changes.

These exclusions are capability boundaries, not silent fallbacks.

## 4. Existing assets and authoritative boundaries

### 4.1 Current vNext assets

The current repository already provides:

- stable `CircuitIR`, instructions, measurements, observables, serialization,
  and content identity in `flagquantum/core/ir.py`;
- target-independent optimization and topology-aware circuit compilation in
  `flagquantum/compiler`;
- local statevector, MPS, tensor-network, and selected gradient executors;
- `fq.Module` with PyTorch-first training behavior;
- a JAX kernel path exposed through `torch.autograd.Function`;
- an internal `ProgramArtifact` candidate envelope;
- a compiler extension boundary that currently consumes and returns `CircuitIR`.

None of these types is to be duplicated under a new name.

### 4.2 Historical research implementation used as evidence

Repository history and the retained workspace copy contain a former private
compiler tree with:

- immutable types, values, operations, blocks, regions, and modules;
- deterministic identities and frozen attributes;
- operation schemas and a fail-closed verifier;
- def-use and qubit-lifetime analyses;
- pass management, static canonicalization, decomposition, placement, routing,
  emitters, target capabilities, TargetIR, and executable artifact prototypes.

It was removed from the current branch because it was disconnected from the
public/default product path. It is evidence for semantics and negative tests,
not a source tree, branch, or class hierarchy to restore. Migration means
re-expressing only required behavior inside current vNext authorities; copying
the former `_compiler`, `QuantumIR`, `TargetIR`, artifact, provider, or runtime
layers is prohibited.

### 4.3 Missing capability

The missing capability is not another static circuit IR. It is the program-level
semantics that preserve tensor values and structured classical control around
quantum effects:

```text
function / call / return
tensor argument / extract / shape
arithmetic / comparison
if / for / while / yield
quantum region / operation / expectation
forward context / VJP
```

## 5. Target architecture

```text
Python source and tensor inputs
           |
           v
Hybrid frontend capture
  AST structure + framework tensor values
           |
           v
Private Program IR
  functions / blocks / regions / typed SSA values / effects
           |
           +--> verifier and capability diagnostics
           |
           v
Partial evaluation and specialization
  static Python values resolved; runtime tensor decisions retained
           |
           v
Quantum-region lowering
  selected operations + late-bound parameter references
           |
           v
Existing CircuitIR and Compiler
           |
           v
Existing Runtime and Simulation
           |
           +--> forward value
           |
           v
Quantum VJP + classical VJP
           |
           v
Gradient in the originating PyTorch graph
```

This is a semantic migration, not a restoration of the historical layer stack:

```text
historical capability                 vNext authority
---------------------                 ---------------
program-level SSA/control/effects  -> flagquantum.compiler._hybrid
static quantum region              -> existing flagquantum.core.ir.CircuitIR
optimization/routing/target emit   -> existing flagquantum.compiler
execution lifecycle                -> existing flagquantum.runtime
numerical state and kernels        -> existing flagquantum.simulation
```

The first phase adds only the missing program-level semantics. A separate
`QuantumIR` or `TargetIR` may be introduced later only if an existing vNext type
cannot express a verified requirement and a new contract is approved.

The architecture has two execution modes:

1. **Path-specialized static region**: evaluate runtime classical branches,
   record the selected quantum operations, lower the resulting region to
   `CircuitIR`, and cache the compiled execution shape.
2. **Structured session**: preserve control-flow boundaries and execute quantum
   regions in sequence. This mode is reserved for later measurement feedback;
   it is not required for the first vertical slice.

## 6. Private Program IR semantic minimum

### 6.1 Types

The first profile requires:

```text
BoolType
IndexType
ScalarType(dtype)
TensorType(dtype, rank, static-or-symbolic dimensions)
QubitRefType
ObservableType
ExpectationType(dtype)
EffectTokenType
```

Tensor values retain shape, dtype, device class, and origin. Conversion to a
Python `float`, `.item()`, NumPy scalar, or detached host value is forbidden for
runtime parameters.

### 6.2 Operations

The first profile requires only the operations used by the golden scenario:

```text
program.func
program.return
tensor.dim
tensor.extract
arith.constant
arith.add
arith.rem
arith.cmp
scf.for
scf.if
scf.yield
quantum.angle_embedding
quantum.rx
quantum.ry
quantum.cx
quantum.expectation
gradient.vjp
```

Operation names are illustrative internal names. They do not create a public
MLIR dialect or compatibility promise.

### 6.3 Values and effects

- Every value has one deterministic definition and typed uses.
- Branch result shapes and types match at joins.
- Loop-carried values have explicit initial, iteration, and result values.
- Quantum operations are ordered through a quantum effect token or an
  equivalent verified linear-state rule.
- A quantum value/effect cannot be duplicated across branches.
- Runtime parameter references remain late-bound through circuit lowering.
- Source locations are diagnostic metadata and do not change semantic identity.

### 6.4 First-profile verifier

The verifier rejects:

- use before definition and duplicate definition;
- operand/result type mismatch;
- branch signature mismatch;
- invalid loop bounds or loop-carried values;
- non-tensor runtime predicates;
- unsupported Python side effects;
- conversion of differentiable tensor parameters to host scalars;
- unsupported quantum operations or observables;
- qubit/effect reuse, loss, or ambiguous branch merge;
- gradients requested through unsupported measurements or dynamic effects;
- implicit backend or gradient-method fallback.

## 7. Frontend capture boundary

### 7.1 Supported Python subset

The initial frontend accepts:

- a decorated function with positional tensor inputs;
- local scalar/tensor assignments;
- `if`/`elif`/`else` with tensor-scalar predicates;
- `for` over `range` and statically ranked tensor axes;
- nested supported control flow;
- supported arithmetic, comparison, indexing, and FlagQuantum calls;
- a single scalar expectation return.

### 7.2 Unsupported subset

The frontend rejects with source-located diagnostics:

- object or container mutation visible outside a region;
- `break`, `continue`, `yield`, exceptions, context managers, comprehensions,
  generators, recursion, and dynamic imports;
- file, process, network, logging, printing, and global-state effects;
- `Tensor.item()`, `float(Tensor)`, NumPy conversion, or detach on a required
  gradient path;
- data-dependent tensor rank or result signature;
- calls without a registered capture rule.

### 7.3 Capture implementation sequence

The first frontend uses Python `ast` to preserve native control-flow structure.
It does not execute runtime tensor predicates during capture. PyTorch FX/Dynamo
integration follows only after Program IR semantics and direct execution are
proven. This avoids making a framework graph a second source of quantum truth.

## 8. Lowering and execution

### 8.1 Partial evaluation

Compile-time constants, static ranges, and source structure are resolved once.
Runtime tensor values remain typed parameters. Constant folding must never read
a runtime tensor value through `.item()`.

### 8.2 Path-specialized lowering

For the initial profile, a runtime invocation evaluates structured classical
predicates and records the selected gate sequence. The trace is lowered into the
existing `CircuitIR` using late-bound tensor expressions or parameter slots.

The cache separates:

```text
program structure identity
input signature identity
selected quantum structure identity
parameter values
```

Parameter values never enter the reusable structure identity unless explicitly
declared static. Branch-dependent structures may produce multiple bounded cache
entries. Cache exhaustion fails or evicts visibly; it must not silently compile
without accounting.

### 8.3 Existing compiler and simulation reuse

The lowered `CircuitIR` passes through the existing compiler. The first runtime
target is local CPU statevector. MPS, tensor network, GPU, JAX, distributed, and
provider paths are excluded until the same scenario is independently accepted
for those modes.

This path is classified as `single_device_fast_path`.

## 9. Gradient contract

### 9.1 Required invariant

For every differentiable runtime input, the compiler preserves this chain:

```text
source tensor
  -> Program IR SSA value
  -> gate parameter slot
  -> quantum expectation
  -> quantum VJP
  -> classical VJP
  -> source tensor gradient
```

No source-to-IR, IR-to-circuit, runtime, result, or framework adapter may detach,
copy through a scalar host value, or replace an unsupported derivative with zero.

### 9.2 Initial method

The first profile computes a VJP for a scalar analytic expectation on a pure
statevector. The implementation may use an existing FlagQuantum reverse/adjoint
executor where its semantic contract matches the selected circuit. Parameter
shift is allowed only as an explicitly selected reference method, never as a
silent performance fallback.

### 9.3 Control-flow differentiation

Reverse execution follows the branch selected by the forward invocation and
reverses loop iterations in execution order. A predicate such as `p > 0` is not
differentiated as a smooth function. Gradients are branchwise and the switching
boundary is reported as non-smooth.

### 9.4 PyTorch integration

After the direct internal path is accepted, the quantum region is registered as
a functional PyTorch custom operation with:

- an explicit schema and no hidden input mutation;
- FakeTensor/meta behavior;
- saved forward context;
- a registered VJP/backward rule;
- `opcheck` and `gradcheck` coverage;
- `torch.compile` full-graph and graph-break tests.

## 10. Golden scenario

The first scenario is semantically equivalent to:

```python
def cost(weights, data):
    angle_embedding(data, wires=range(4))
    for layer in weights:
        for j, p in enumerate(layer):
            if p > 0:
                rx(p, wire=j)
            elif p < 0:
                ry(p, wire=j)
        for j in range(4):
            cx(j, (j + 1) % 4)
    return expectation(z(0) + z(3))
```

Acceptance requires:

- positive, negative, zero, and mixed branch patterns;
- repeated invocation with different values and the same signature;
- branch structure cache identity and bounded cache behavior;
- deterministic IR identity independent of source-location movement;
- forward agreement with an explicit hand-built `CircuitIR` reference;
- gradient agreement with central finite differences away from branch boundaries;
- gradient agreement with the existing uncompiled FlagQuantum path where
  equivalent;
- optimization steps that decrease a seeded differentiable objective;
- explicit rejection at `p == 0` when a test asks for a smoothness guarantee;
- no change in existing non-hybrid `fq.run`, `fq.plan`, or `fq.Module` behavior.

## 11. Execution phases and gates

### Phase 0 — reconcile and freeze the vertical slice

Deliverables:

- current/removed implementation inventory;
- authoritative ownership map;
- exact golden scenario and unsupported boundary;
- reuse/subtraction decision for each old compiler component;
- baseline tests for existing circuit compile, statevector forward, and gradient;
- API impact statement.

Exit gate:

- the current compiler-extension/public API work is committed or otherwise
  cleared from the integration worktree;
- Integration approves the private cross-domain contract;
- no public schema or root export is changed;
- Compiler and Runtime owners agree on the `CircuitIR` handoff and VJP boundary.

### Phase 1 — private Program IR semantic core

Compiler-owned files only. Deliver immutable values, operations, blocks,
regions, types, schemas, deterministic identity, source diagnostics, and the
first-profile verifier.

Exit gate:

- positive construction tests;
- negative verifier tests for every declared invariant;
- deterministic identity tests;
- contributor golden path under ten minutes;
- no import from Runtime, Simulation, provider, or ecosystem modules.

### Phase 2 — restricted Python capture

Deliver AST capture for the supported subset and source-located rejection for
the unsupported subset.

Exit gate:

- golden source captures without evaluating runtime predicates;
- all prohibited constructs fail closed;
- equivalent formatting and local renaming preserve semantic identity where
  names are not semantically observable;
- capture does not modify globals or user tensors.

### Phase 3 — path specialization and CircuitIR lowering

Deliver partial evaluation, runtime branch selection, gate tracing, parameter
slots, structure cache, and lowering to the existing `CircuitIR`.

Exit gate:

- all golden branch patterns match hand-built `CircuitIR` references;
- parameter values remain differentiable tensors;
- structure identities exclude late-bound values;
- existing compiler passes accept the lowered output;
- unsupported dynamic effects fail before simulation.

### Phase 4 — local CPU forward vertical slice

Contract-first cross-domain work. Runtime accepts the compiler-produced existing
artifact at an internal boundary and invokes existing statevector simulation.

Exit gate:

- scalar expectation matches the explicit reference for all branch patterns;
- errors retain compiler/runtime/simulation ownership;
- no default path or backend fallback changes;
- local fast-path regression budget is met when hybrid compilation is unused.

### Phase 5 — first-order VJP vertical slice

Deliver quantum VJP, classical chain rule, branch/loop reverse semantics, and
finite-difference differential tests.

Exit gate:

- gradients match numerical references away from branch boundaries;
- zero/detached gradients fail tests;
- gradient method and limitations are observable;
- a seeded optimizer decreases the objective over multiple steps.

### Phase 6 — PyTorch graph compilation

Deliver the functional custom operation, Autograd registration, FakeTensor
contract, and `torch.compile` integration.

Exit gate:

- eager and compiled forward/gradient parity;
- `opcheck` and `gradcheck` pass;
- no unexpected graph break in the supported scenario;
- compile latency, cache hit, steady-state step time, and memory are reported
  separately without a performance claim unless gates are met.

### Phase 7 — dynamic measurement session

This phase requires a separately approved Runtime/Simulation contract. Add
mid-circuit measurement values, stateful sessions, conditional continuation,
shot semantics, and stochastic-gradient policy.

It is not a prerequisite for claiming the bounded classical-data-dependent
hybrid control-flow profile from Phases 0-6.

### Phase 8 — parameterized bounded dynamic session

Extend the private Phase 7 session with non-trainable scalar, index, and bool
runtime inputs. Specialize positive bounded `range` loops, lower RX/RY values
through explicit Core `Parameter` slots, and keep parameter values outside the
template identity.

Exit gate:

- equal structure with different parameter values has one template identity;
- bound values preserve the original scalar tensor objects;
- loop bounds are visible and limited by a configurable unroll ceiling;
- eager dynamic shots respond to parameter changes under deterministic oracles;
- trainable inputs, unbound templates, tensor inputs, and loop measurements
  fail closed;
- no finite-shot gradient, graph-compilation, accelerator, distributed, or
  performance claim is introduced.

### Phase 9 — explicit loop-carried classical state

Extend restricted capture so a bounded `for` loop may update an outer scalar,
index, or bool local and make the final value available after the loop. Capture
must represent every carried value explicitly as an `scf.for` operand, body
argument, yield operand, and operation result. Lowering continues to specialize
and unroll the loop; Runtime and Simulation contracts do not change.

The accepted source profile is deliberately narrow: assignments must target one
existing local name directly in the loop body and preserve its IR type. Tensor
state, loop-target shadowing, writes from nested loops or branches, augmented
assignment, measurement inside loops, and dynamic finite-shot gradients remain
unsupported and fail closed.

Exit gate:

- scalar and index state are visible in the verified `scf.for` signature;
- every iteration consumes the prior carried value and yields its successor;
- post-loop quantum operations consume the explicit loop result;
- dynamic execution observes the expected accumulated angles and wire choices;
- invalid carry types, target shadowing, nested/conditional writes, and loop
  measurement fail closed;
- no Runtime, Simulation, public API, default-path, or performance change is
  introduced.

### Phase 10 — explicit branch-carried classical state

Extend restricted capture so an `if/else` may update existing scalar, index,
or bool locals and expose the selected values after the branch. Every carried
name is represented explicitly as an `scf.if` operand, an argument of both
regions, a yield from both regions, and an `scf.if` result. If only one side
assigns a name, the other side yields its unchanged block argument.

Dynamic lowering accepts this profile only when the predicate is resolved by
specialization. A predicate produced by mid-circuit measurement cannot export
classical state into the static Core `CircuitIR`; that case fails before either
branch is lowered. Tensor state, nested branch/loop writes, augmented
assignment, finite-shot gradients, and general runtime classical expressions
remain unsupported.

Exit gate:

- scalar, index, bool, and effect values have matching branch signatures;
- one-sided assignments have explicit unchanged-value pass-through;
- post-branch gates and measurement wires consume `scf.if` results;
- both specialized paths produce deterministic expected circuits and outcomes;
- tensor, nested, and measurement-dependent carried state fail closed;
- Runtime, Simulation, public API, default path, and performance claims remain
  unchanged.

### Phase 11 — nested structured state propagation

Generalize carried-name discovery across nested supported `if` and `for`
statements. If an inner region updates an outer scalar, index, or bool, every
enclosing structured operation must carry that name explicitly. This creates a
continuous SSA chain from the entry value through each region argument, yield,
and result to the post-structure consumer.

The accepted vertical slice covers an outer bounded loop containing branches
that update angle, wire, and selector state, and an outer branch containing a
bounded loop that accumulates an angle. Lowering still specializes all
classical predicates and bounded trip counts before handing one bound
`CircuitIR` to Runtime.

Exit gate:

- recursive discovery is deterministic and limited to supported statements;
- outer and inner region signatures explicitly contain the same required state;
- loop-then-branch and branch-then-loop paths preserve Python value flow;
- cumulative nested unrolling remains subject to the configured ceiling;
- measurement-dependent state escape, tensor state, and loop measurement fail
  closed;
- Runtime, Simulation, public API, default path, and performance claims remain
  unchanged.

### Phase 12 — measurement boolean predicates

Extend the Program IR with boolean `not` and `and`, and permit equality or
inequality between one measurement boolean and a bool constant. Compiler keeps
these values symbolic and lowers the accepted subset to the existing sorted
`conditions` tuple on Core instructions. Runtime already interprets that tuple
as a conjunction for both reference and batched trajectory execution, so no new
classical runtime or serialized Core type is introduced.

The first profile supports direct measurement literals, one-literal negation,
bool comparison, and conjunctions of distinct literals. A conjunction may
guard quantum work only in its true branch because its false complement is a
disjunction. Boolean `or`, negation of a conjunction, measurement-to-measurement
comparison, inline measurement calls inside a boolean expression, and
measurement-dependent carried state remain unsupported.

Exit gate:

- two top-level measurements produce independently addressable bool SSA values;
- `first and not second` lowers to conditions `((0, 1), (1, 0))`;
- bool comparison lowers true and false branches to complementary conditions;
- reference and batched trajectories agree shot-wise with the predicate;
- unsupported non-conjunctive expressions fail before Runtime execution;
- Runtime, Simulation, public API, default path, and performance claims remain
  unchanged.

### Phase 13 — bounded canonical measurement predicates

Extend Program IR with boolean `or` and represent symbolic measurement
predicates as canonical DNF. Conjunction distributes over clauses,
disjunction unions clauses, negation applies bounded De Morgan expansion, and
measurement equality/inequality lowers to XNOR/XOR. Contradictory, duplicate,
and subsumed clauses are removed with deterministic ordering.

One conjunction continues to use Core `conditions`; predicates requiring more
than one clause use private `condition_clauses`. Local reference and batched
trajectory execution evaluates these clauses as OR-of-AND. The configurable
default limit is 64 clauses. Unsupported expansion, conditional measurement,
measurement-dependent carried state, and provider export of complex clauses
fail closed.

Exit gate:

- disjunction and negated conjunction agree with recorded bits shot by shot;
- equality and inequality between measurements implement XNOR and XOR;
- general quantum true/false branches receive exact complementary predicates;
- canonicalization is stable and removes contradictions and subsumption;
- the clause ceiling and malformed runtime metadata fail explicitly;
- simple conjunction metadata remains backward compatible;
- public API, default path, stochastic-gradient, and performance claims remain
  unchanged.

### Phase 14 — measurement-dependent SSA value merging

Represent scalar, index, and bool values leaving a measurement-dependent
branch as bounded condition-partitioned SSA cases. Classical operations
distribute over these cases, while quantum consumers split into mutually
exclusive conditioned instructions. Equal cases are coalesced where identity
or primitive equality is unambiguous.

The representation is compiler-private and lowers back into existing Core
instructions plus Phase 13 condition metadata. It therefore introduces no
public classical-store abstraction and no Core schema change. Static bounded
loops can carry conditional values, but measurement-dependent loop bounds,
conditional measurement, dynamic program returns, and stochastic gradients
remain rejected.

Exit gate:

- a measurement branch exports scalar, index, and bool values together;
- post-branch arithmetic preserves the correct per-shot scalar parameter;
- conditional wire selection splits gates under exclusive predicates;
- a carried bool controls a later structured branch exactly;
- reference and batched trajectories agree with recorded measurement bits;
- case growth is bounded by the configured predicate ceiling;
- public API, Core schema, default path, and performance claims remain
  unchanged.

### Phase 15 — fixed-round syndrome measurement and feedback

Permit measurements and unconditional ancilla reset inside statically bounded
loops. Lowering unrolls the rounds, allocates dense classical bits for each
measurement, carries the latest syndrome through loop SSA, and emits immediate
conditioned correction gates without evaluating measurements in Compiler.

The first QEC-oriented vertical slice injects a deterministic data error,
extracts an ancilla syndrome for three rounds, corrects after the first
detection, resets the ancilla every round, and verifies that subsequent
syndromes remain clear. Reference and batched Runtime strategies must agree.

Exit gate:

- loop measurement order maps deterministically to dense classical bits;
- a loop-carried syndrome may be returned after the final round;
- measurement feedback changes the data qubit in the same round;
- unconditional reset returns the ancilla to zero before reuse;
- unroll, measurement, and predicate expansion budgets fail closed;
- conditional measurement/reset and dynamic termination remain unsupported;
- no decoder, threshold, provider, gradient, capacity, or performance claim is
  made.

### Phase 16 — QEC-owned repetition-code memory workflow

Create an experimental `flagquantum.qec` domain without adding stable root
exports or moving generic dynamic control out of Compiler and Runtime. Compose
the Phase 15 machinery into a fixed-round, three-data-qubit repetition-code
memory experiment with two adjacent parity checks and reusable ancillas.

Return typed shot-resolved syndrome, detection-event, decoder-decision, final
data, and logical-result records. Provide a small decoder protocol and lookup
reference. The replaceable decoder initially analyzes recorded syndromes after
execution; the compiled lookup remains the policy that performs feedback.

Exit gate:

- zero-error logical-zero memory is preserved;
- one X error on each data wire produces the expected first syndrome;
- compiled same-round lookup feedback restores the data register;
- later syndromes clear and temporal detection events are explicit;
- reference and batched trajectory strategies agree;
- analysis decoder replacement is structurally verified;
- malformed inputs and inconsistent QEC records fail closed;
- no stable-root, realistic-noise, real-time-decoder, threshold,
  fault-tolerance, provider, gradient, capacity, or performance claim is made.

### Phase 17 — timed errors, history decoding, and offline Pauli frames

Replace the initial single-wire injection shortcut with a bounded canonical
schedule of deterministic X-error events located by round and data wire.
Specialize those events into the unrolled program before each round's parity
checks. Keep this test instrumentation QEC-owned and leave generic dynamic
control with Compiler and Runtime.

Make the decoder consume the complete ordered syndrome history and return a
typed decode result containing correction recommendations and a parity-reduced
Pauli frame. Separate immediate compiled lookup feedback from an offline mode
that performs no physical feedback and applies the frame only to final readout.
Record actual feedback independently from decoder recommendations.

Exit gate:

- the schedule is canonical, bounded, and rejects ambiguous duplicate events;
- every data wire may receive one X error in any verified round;
- compiled feedback and offline frame correction both restore single errors;
- syndrome onset and feedback clearance appear as detection events;
- two same-round errors produce an explicit logical failure rather than a
  success claim;
- malformed histories, decode results, frames, and result records fail closed;
- stochastic/measurement noise, real-time callbacks, threshold, suppression,
  general-code, hardware, gradient, capacity, and performance remain excluded.

### Phase 18 — bounded stochastic noise in dynamic execution

Connect the existing `NoiseModel` authority to local dynamic trajectories.
Runtime owns seeded random-stream lifecycle, channel placement after matching
executed gates, true/observed measurement separation, feedback from observed
bits, and event statistics. Simulation supplies the numerical sampling kernels.

Limit the first profile to independent one-wire bit-flip channels and
independent readout confusion. Apply readout confusion to explicit measurements
and final sampling while collapsing the physical state on the true bit. Reject
general Kraus channels, correlated readout, device-timing profiles, reset noise,
and noisy gradients.

Map the profile onto the repetition-code check circuit and provide typed
finite-shot sweep points. State the circuit-location asymmetry explicitly and
do not infer logical suppression or a threshold from the sweep.

Exit gate:

- probability-one bit flips and readout errors have deterministic oracles;
- gate noise applies only to shots on which a conditional gate executes;
- seeded noisy execution is reproducible within each strategy;
- noise identity, opportunities, realized flips, and readout errors are visible;
- QEC stochastic runs preserve syndrome, feedback, decode, and logical records;
- finite-shot sweeps are observations, not calibrated-device predictions;
- unsupported noise forms fail before execution;
- public-root, provider, gradient, threshold, scale, and performance claims
  remain unchanged.

### Phase 19 — Runtime decoder feedback and Pauli-frame evolution

Add private measurement decision points to the local trajectory executor.
Runtime records true and observed measurement values, calls one bounded
controller after each declared point, executes a physical X or updates an X
Pauli frame, and records the resulting per-shot decision trace. Keep the
contract framework-neutral and keep QEC concepts in the QEC adapter.

Add a `StreamingDecoder` for the repetition-code reference. It consumes the
complete syndrome history available each round. Provide separate Runtime
physical-correction and Runtime Pauli-frame modes; frame state adjusts later
syndrome interpretation and final readout without modifying quantum state.

Exit gate:

- replacing the streaming decoder changes continued execution and its trace;
- compiled lookup, Runtime physical feedback, and Runtime frame feedback agree
  on bounded single-error logical outcomes;
- traces separate true bits, observed bits, actions, and frame evolution;
- decision points and actions outside the declared plan fail closed;
- batched feedback, stable plugin publication, provider/hard-real-time control,
  general codes, gradients, suppression, thresholds, scale, and performance
  remain unsupported.

### Phase 20 — detection-event temporal decoding

Add a bounded repetition-code streaming decoder that distinguishes persistent
data syndromes from isolated readout excursions using consecutive-round
detection events. Require two matching non-zero syndrome rounds before issuing
physical or frame feedback, and reject inconsistent event histories.

Exit gate:

- a persistent syndrome is confirmed and corrected on its second observation;
- an isolated syndrome excursion and matching return event cause no action;
- confirmable single-data errors work through physical and frame policies;
- a terminal-round error remains visible and unconfirmed;
- a seeded readout-noise profile records fewer spurious actions than immediate
  lookup without promoting that observation to a suppression claim;
- maximum-likelihood decoding, arbitrary measurement-error tolerance, general
  codes, hardware timing, thresholds, scale, and performance remain excluded.

### Phase 21 — verified Program IR normalization

Insert one private, bounded transformation stage after Program IR verification
and before both static specialization and dynamic lowering. Keep the same
`HybridProgram` type across the boundary. Compute constant and SSA-use facts,
fold supported constant-only arithmetic, and remove unused constants without
removing operations that may fail at runtime.

Exit gate:

- every input and pass result is verified and invalid pass output fails closed;
- the pipeline is deterministic, idempotent, and capped at 32 passes;
- static and dynamic lowering match an explicit unoptimized differential oracle;
- source and optimized identities plus per-pass operation counts are retained;
- tensor parameter bindings and autograd edges survive normalization;
- no target IR, target dialect, stable pass extension API, or performance claim
  is introduced.

### Phase 22 — structured-control-flow simplification

Extend the verified normalization pipeline with one transformation over the
existing structured Program IR. Inline only an `scf.if` whose Boolean predicate
is known by constant analysis, and remove only an `scf.for` whose constant
bounds prove it has zero iterations. Rewire classical carried values and the
linear quantum effect through the selected yield or initial loop operands.

Exit gate:

- selected branches preserve their quantum operation order and carried values;
- unselected branch operations are absent from the optimized program;
- empty loop bodies are absent and loop-carried values retain initial values;
- measurement-dependent and otherwise unresolved control remains represented;
- static and dynamic lowering match their unoptimized differential oracles;
- the resulting Program IR passes the ordinary SSA/type/effect verifier;
- no general unrolling, loop motion, target IR, public API, or performance claim
  is introduced.

### Phase 23 — bounded constant-loop unrolling

Unroll a direct entry-block `scf.for` only when all bounds are compile-time
integers, the step is nonzero, the loop has at most eight iterations, and total
expansion stays within 256 operations. Clone fresh SSA definitions and an
explicit induction constant per iteration, then chain every classical carried
value and the linear quantum effect through the cloned yields.

Exit gate:

- a three-iteration parameterized quantum loop lowers identically with the pass
  enabled and disabled;
- parameter values and their autograd edges survive cloning;
- every cloned definition remains unique and the transformed IR verifies;
- normalization is deterministic and idempotent after expansion;
- compile-time-expanded iterations still count against the caller's existing
  unroll limit in static and dynamic lowering;
- iteration or operation budget overflow leaves the structured loop intact;
- nested/general unrolling, target scheduling, public API, and performance
  claims remain excluded.

### Phase 24 — pass audit and differential verification

Split the private optimizer by responsibility into analysis, SSA rewrite,
concrete transformation, and pipeline/audit modules. Give every concrete pass
an immutable outcome with non-negative statistics and deterministic remarks for
preserved loops. Keep audit evidence outside Program IR identity.

Build a fixed-seed differential corpus that exercises constant branches,
bounded loops, carried values, quantum-effect ordering, and parameters. Compare
optimization enabled and disabled at the `CircuitIR`, statevector result, and
adjoint-VJP levels. Re-run optimization to prove fixed-point behavior. Record
negative-step loops as an equal fail-closed boundary until loop bounds gain a
separate signed-integer type.

Exit gate:

- four local modules have one documented optimizer responsibility each;
- every default pass reports actual transformation counts;
- budget-preserved loops report deterministic identities and reasons;
- pass statistics and remarks are immutable and identity-neutral;
- at least 12 fixed seeds agree on structure, parameters, expectation, and VJP;
- every optimized seeded program is a fixed point on a second pass;
- negative-step behavior agrees with optimization enabled and disabled;
- no new optimization, target IR, public API, or performance claim is added.

## 12. File and team ownership plan

Expected Compiler-owned implementation (files are added only when their phase
needs them):

```text
flagquantum/compiler/_hybrid/
  README.md
  model.py
  schemas.py
  verifier.py
  capture.py
  specialize.py
  lowering.py
  diagnostics.py
```

Expected shared tests:

```text
tests/hybrid_compiler/
  test_program_ir.py
  test_verifier_negative.py
  test_capture.py
  test_lowering.py
  test_forward_vertical.py
  test_gradient_vertical.py
  test_torch_compile.py
```

Runtime and Simulation changes must be separate domain-owned changes after an
Integration-owned contract is approved. Core types or public APIs are not to be
changed by a team branch.

## 13. Verification ladder

For each phase:

1. run the smallest new focused unit tests;
2. run existing compiler tests;
3. run existing statevector and gradient tests when the handoff is touched;
4. run `python tools/check_team_scope.py` for the owning team;
5. run `python tools/check_architecture.py`;
6. run `python tools/ci_tier.py pr-default`;
7. run `python tools/ci_tier.py pr-runtime` for vertical execution changes;
8. compare the protected public API snapshot without regenerating it.

All commands must use the repository-supported Python environment and bounded
timeouts. Empty test selections are not evidence.

## 14. Performance evaluation

The first performance report separates:

- source capture time;
- Program IR verification and lowering time;
- branch specialization time;
- structure-cache hit rate;
- circuit compile time;
- statevector forward time;
- VJP time;
- end-to-end eager step time;
- end-to-end compiled steady-state step time;
- peak host and device memory.

Compilation benefit is reported only after amortization:

```text
break_even_steps = compile_time / (eager_step_time - compiled_step_time)
```

No kernel, distributed, or capacity benefit may be attributed to hybrid
compilation without direct evidence.

## 15. Risks and controls

| Risk | Control |
| --- | --- |
| Another disconnected compiler tree | Every phase ends in the same golden vertical slice; unused abstractions are removed |
| Public API drift | Private namespace first; separate proposal before any root export |
| Gradient detachment | SSA origin tracking, forbidden scalar extraction, VJP and optimizer tests |
| Branch-cache explosion | Bounded cache, visible eviction/failure, structure/value identity separation |
| Python interpreter overhead | Region batching first, PyTorch compilation after semantics pass |
| PyTorch becomes the quantum truth | Program IR owns quantum semantics; PyTorch owns classical graph execution |
| Silent unsupported behavior | Capability verifier and typed diagnostics before execution |
| Local-path regression | Hybrid path is opt-in and benchmarked separately |
| Old code volume returns | Migrate behavior by need; do not copy provider/deployment machinery |

## 16. Stop and review conditions

Stop implementation and return to Integration review if:

- a protected public API or serialized Core schema must change;
- the vertical slice requires modifications in four or more primary domains;
- runtime tensor values cannot remain attached through current `CircuitIR`;
- the existing compiler-extension contract conflicts with hybrid artifacts;
- the local CPU path slows when hybrid compilation is unused;
- an unsupported feature would require silent fallback;
- a second source of truth for circuits, results, backends, or plugins appears.

## 17. Current progress

- [x] Workspace and repository-rule inspection
- [x] Current static compiler and extension-boundary inspection
- [x] Historical multi-level IR implementation located
- [x] Historical core IR tests sampled successfully
- [x] Static versus dynamic Program IR boundary identified
- [x] Detailed execution plan recorded
- [x] Existing compiler-extension/public API worktree changes cleared
- [x] Phase 0 private semantic contract approved conditionally
- [x] Machine-readable entry gates added
- [x] Whole-tree restoration explicitly rejected
- [x] Migration mapped onto current vNext authorities
- [x] Integration team-scope preflight passed for the Phase 1 slice
- [x] Phase 1 private semantic slice implemented and verified
- [x] Phase 2 private capture contract authorized
- [x] Phase 2 restricted Python capture implemented and verified
- [x] Phase 3 private specialization/lowering contract authorized
- [x] Phase 3 path specialization and `CircuitIR` lowering verified
- [x] Phase 4 existing `CircuitIR`/`MeasurementNode` handoff contract authorized
- [x] Phase 4 local CPU forward vertical slice verified
- [x] Phase 5 local CPU VJP contract authorized
- [x] Phase 5 branchwise adjoint VJP vertical slice verified
- [x] Phase 6 functional PyTorch custom-op contract authorized
- [x] Phase 6 FakeTensor, Autograd, opcheck, gradcheck, and fullgraph path verified
- [x] Phase 7 Runtime/Simulation contract authorized for a bounded profile
- [x] Phase 7 measurement-value, conditional-session, shot, and gradient-policy slice verified
- [x] Phase 8 parameter-slot and bounded-loop contract authorized
- [x] Phase 8 non-trainable parameterized dynamic-session slice verified
- [x] Phase 9 explicit loop-carried classical-state contract authorized
- [x] Phase 9 scalar/index carry capture and dynamic execution verified
- [x] Phase 10 explicit branch-carried classical-state contract authorized
- [x] Phase 10 scalar/index/bool merge and dynamic execution verified
- [x] Phase 11 nested structured-state propagation contract authorized
- [x] Phase 11 loop/branch and branch/loop value-flow slices verified
- [x] Phase 12 measurement-boolean predicate contract authorized
- [x] Phase 12 negation, comparison, and conjunctive feedback verified
- [x] Phase 13 bounded canonical predicate contract authorized
- [x] Phase 13 disjunction, complement, XOR/XNOR, and exact quantum branches verified
- [x] Phase 14 measurement-dependent SSA merge contract authorized
- [x] Phase 14 scalar/index/bool value partitioning and consumer splitting verified
- [x] Phase 15 fixed-round syndrome-feedback contract authorized
- [x] Phase 15 loop measurement, immediate correction, and ancilla reset verified
- [x] Phase 16 QEC domain and repetition-code memory workflow verified
- [x] Phase 17 timed errors, history decoding, and offline Pauli frames verified
- [x] Phase 18 bounded dynamic noise and QEC finite-shot sweeps verified
- [x] Phase 19 Runtime decoder feedback and Pauli-frame evolution verified
- [x] Phase 20 detection-event temporal decoding verified
- [x] Phase 21 verified Program IR normalization implemented and verified
- [x] Phase 22 structured-control-flow simplification implemented and verified
- [x] Phase 23 bounded constant-loop unrolling implemented and verified
- [x] Phase 24 pass audit and differential verification implemented and verified
