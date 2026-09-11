# Multi-Level IR and Quantum Compilation Infrastructure

Status: historical design baseline and long-term architecture direction.
For current source paths and implemented compiler behavior, read the
[implementation map](../development/IR_IMPLEMENTATION_STATUS.md). Statements
such as `fq.compile` being absent describe the recorded baseline, not the
current public API. Proposed interfaces below are not implementation contracts.

Scope: compilation, Runtime, interoperability, and QPU deployment.

This design does not change public `CircuitIR` schema 1.0, Stable Core, or released
contracts. Names are internal placeholders unless a separate API proposal approves them.

## 1. Purpose

Define the evolution from circuit exchange to compilation infrastructure: current
CircuitIR responsibilities, ProgramIR/QuantumIR/TargetIR, OpenQASM 3/QIR/QCIS/local
execution, compiler/Runtime separation, compatible migration, semantic/performance/
reproducibility/legality verification, and when Python versus C++/MLIR is appropriate.

## 2. Recorded Baseline

### 2.1 Existing Capabilities

Circuit.to_ir() generates/caches CircuitIR. Schema 1.0 contains instructions,
observables, measurements, dtype, shape, metadata; it supports validation,
deterministic JSON, and hashing. Compiler/planner/drawer/local/distributed execution
consume it. Existing transformations include cancellation, rotation merging,
scheduling, routing, post-routing optimization. Deployment compiles to QASM 2/3 or
QCIS. Dynamic construction, local simulation, batched trajectories, capability
assessment, and QASM 3 have experimental paths. CircuitIR is release-certified and
protected.

Implementation references: `flagquantum/core/ir.py`, `flagquantum/circuit.py`,
`flagquantum/compiler/pipeline.py`, `flagquantum/compiler/routing.py`,
`flagquantum/runtime/execution.py`, `flagquantum/runtime/dynamic/`,
`flagquantum/deployment/cloud.py`, `flagquantum/compiler/openqasm.py`,
`flagquantum/compiler/qcis.py`.

### 2.2 Limitations

CircuitIR is a flat instruction sequence suited to exchange, simulation, and
training. It lacks typed functions/kernels/scopes/blocks/calls, shared classical/
quantum/measurement def-use, unified loops/branches/feedback, dynamic integration
with ordinary run, typed logical/physical layout/timing, shared target legality,
format-neutral artifacts, pass preconditions/preservation, and executable ABI.
Current artifacts center on QASM with QCIS in metadata. Address these behind
CircuitIR rather than breaking it.

## 3. Architecture Decision

```text
fq.Circuit / fq.Module / import adapters
                    |
                    v
       Public CircuitIR (schema 1.0)
        stable exchange contract
                    |
           import + legalization
                    v
              ProgramIR
     functions / classical control / calls
                    |
          partial evaluation + lowering
                    v
              QuantumIR
    qubits / gates / measurements / results
                    |
       target selection + optimization
                    v
               TargetIR
 physical layout / native gates / scheduling
                    |
       code generation / plan generation
          +---------+----------+----------+
          |         |          |          |
          v         v          v          v
       QASM 3      QIR        QCIS    Runtime Plan
          |         |          |          |
          +---------+----------+----------+
                    |
                    v
             ExecutableArtifact
                    |
                    v
              RuntimeAdapter
       local simulator / cloud QPU / controller
```

Keep CircuitIR public and serializable; initially keep all three richer IRs
internal. QASM/QIR/QCIS are target formats, not authorities. Separate compilation
from submission; reject target incompatibility before submission; fix structure
early and bind numbers late. Use differential migration with bounded coexistence.
Start in Python; evaluate native infrastructure only against demonstrated needs.

## 4. Principles

Public exchange remains CircuitIR; users need not learn dialects/passes. Move
measurement results, conditions, logical/physical qubits, binding phases, gates,
durations/schedules, observables/requests, and result schemas into typed semantics.
Metadata carries extensions/debugging/nonsemantic provenance only.

Passes declare preservation of state/density, measurement distributions,
expectations, gradients, approximation budgets, ordering, and feedback. Record
approximation/noise/fallback/reordering explicitly. Reject unsupported measurement,
reset, realtime control, gates, qubit limits, and QIR profiles at the earliest
knowable stage. Preserve local fast paths through caching, lazy compilation,
and bypass; full QPU targeting is not mandatory for local circuits.

## 5. IR Levels

### 5.1 Public CircuitIR

Own stable exchange, JSON/content addressing, REST/MCP/task boundaries, compatibility
between Circuit/adapters/executors, and current static training/deployment inputs.
Do not turn it into full SSA, general classical control, native timing, provider
executables, or pulse programs. Preserve `IR_VERSION = "1.0"`, avoid root internal
exports, provide importers and lossless or explicitly bounded v1 round trips.
Reject unsupported rich-to-v1 conversion rather than dropping dynamic semantics.

### 5.2 ProgramIR

Represent hybrid programs with modules, functions/kernels, typed arguments/results,
scopes, blocks/branches/loops/calls, compile constants, runtime parameters,
measurement-produced values, host-only/controller-eligible regions, and source
locations/diagnostics.

```text
func active_reset(theta: angle) -> bit {
    q = quantum.alloc
    quantum.rx(theta, q)
    m = quantum.measure(q)
    if m {
        quantum.x(q)
    }
    return quantum.measure(q)
}
```

Verify symbols/scopes, types, terminators, returns, quantum/result usage, and
whether controller regions contain operations the target cannot execute in real time.

### 5.3 QuantumIR

Hardware-independent quantum semantics remove statically evaluable classical
structure while retaining measurement-dependent control where needed.

```text
QubitType
QubitRegisterType
MeasurementResultType
AngleType
ParameterType
ObservableType
SampleResultType
```

```text
alloc / release
gate / controlled_gate
measure / reset
call
expectation / sample
barrier
conditional region
```

Qubits cannot be copied; verify lifetimes, explicit measurement def-use, legal
conditions, static/late parameters, typed observables/requests, shared static/
adaptive representation. Immutable Python nodes, value IDs, block arguments, and
def-use indexes can implement bounded SSA without a general-language compiler.

#### 5.3.1 References and Linear Values

```text
ProgramIR: reference-oriented
  Imperative builders, scopes, calls, and classical control

QuantumIR: value/linear-oriented
  Explicit quantum dataflow for verification, rewrites, scheduling, lowering
```

ProgramIR references identify controlled access to resources, not copyable states.
Ownership analysis tracks aliases and rejects independent writable duplicates.
Lowering consumes the current value and creates its successor:

```text
%q1 = quantum.h %q0
%q2, %q3 = quantum.cx %q1, %q_other
%q4, %m = quantum.measure %q2
%q5 = quantum.reset %q4
quantum.release %q5
```

Values identify resource state at a program point, not copies of statevectors.
Measurement's reusable-qubit result is explicit; release consumes the final value.
At joins, every predecessor contributes exactly one live value for the same
resource. Reject duplicate consumers, stale values after gates, use after release,
asymmetric release/allocation without valid joins, merging different resources,
nonlinear containers, and ambiguous function return/capture ownership.
TargetIR may map verified values to physical references without restoring implicit
copying. Circuit's imperative user experience remains unchanged.

#### 5.3.2 Ancillas

Ancilla is a resource role, not another physical qubit type. Users can reserve
ordinary Circuit wires manually and use DynamicCircuit measurement/reset for some
reuse. v1 has fixed n_wires and integer references, without ancilla identity,
ownership, allocation, clean/dirty requirements, or lifetime semantics. Routing
cannot use unmapped physical qubits outside CircuitIR; it must reject and request
explicit layout/lowering. Qiskit named/multiple/aliased registers flatten to stable
indices; AncillaRegister qubits survive, but their role is not stable v1 semantics.

Public claims must distinguish manual ancilla use from unavailable compiler-managed
allocation/reuse/release/verification. The design distinguishes data qubits,
user ancillas, compiler-created temporaries, and unmapped physical resources.
Add alloc/release, clean `|0>` and dirty-restoration pre/postconditions, release/
join/ownership checks, constrained allocation for decomposition/QEC/routing,
capability-bound measure/reset/reuse, hidden internal ancilla measurements except
explicit debugging, and reports of logical_qubits, ancilla_qubits,
peak_physical_qubits, source, blockers.

Do not prematurely publish `fq.AncillaQubit`. Keep integer-qubit usage while
internal semantics mature; consider public syntax only after at least two release
cycles and interoperability evidence through a separate proposal.

### 5.4 TargetIR

Represent target identity/capability hash, logical-physical layout, physical
qubits, native/calibrated gates, coupling, durations, delays/barriers/boxes/alignment,
measurement groups, controller instructions, conflicts, ordering, unresolved
parameters. Verify supported operations, connectivity, dynamic controls, valid
resource schedules, qubit/result/shot/parameter limits, and accepted output profiles.

### 5.5 Program Versus Request

| Concept | Owner | Meaning |
| --- | --- | --- |
| Gates/alloc/release/reset | ProgramIR/QuantumIR | Program semantics |
| Mid-circuit measurement | ProgramIR/QuantumIR | Feeds later control |
| Conditions/loops | ProgramIR | Static evaluation, dynamic lowering |
| Explicit terminal measurement | QuantumIR | Part of the program |
| Observable definition | Typed domain object | Referenced by requests |
| Expectation/sample/state | ExecutionRequest | Requested output, not source rewrite |
| Shots/seed/batch | ExecutionOptions | Outside program identity |
| Backend/device/routing | CompilationPolicy | Affects lowering/artifact identity |
| Explicit noise channel | QuantumIR | Semantic only when written into program |
| Noise model/fallback | Compilation/ExecutionPolicy | Record lowering explicitly |
| Gradient request/method | DifferentiationRequest | Adjoint/shift/native plans |
| Credentials/quota/queue | Runtime | Outside IR, payloads, model context |
| Layout/native gates | TargetIR | Target compilation result |

Import existing observables/measurements into separate internal semantics/requests
without changing public schema. Explicit run requests obey approved conflict rules,
not importer-specific overrides.

```text
program_identity
  core program semantics + static parameters

compilation_identity
  program_identity + target + compiler + pipeline + capability snapshot

execution_identity
  artifact identity + runtime parameters + shots + seed + execution policy
```

Shots changes should not reroute; target/pipeline changes must invalidate artifacts.

## 6. Pass Infrastructure

### 6.1 Contracts

```text
name
input IR level
output IR level
required analyses
preserved analyses
semantic properties preserved
possible diagnostics
determinism guarantee
```

```python
class CompilerPass(Protocol):
    name: str
    input_level: IRLevel
    output_level: IRLevel

    def run(
        self,
        module: IRModule,
        context: CompilationContext,
    ) -> IRModule: ...
```

CompilationContext carries capabilities, pipeline options, diagnostics, source maps,
seed, compiler version, cache policy. No undeclared environment-driven semantics.

### 6.2 Pipeline

```text
Import CircuitIR/OpenQASM
  -> structural verification
  -> type and lifetime verification
  -> canonicalization
  -> constant propagation
  -> static loop unrolling
  -> dead branch elimination
  -> function/gate inlining
  -> symbolic parameter simplification
  -> ProgramIR-to-QuantumIR lowering
  -> gate canonicalization
  -> high-level gate decomposition
  -> observable/measurement lowering
  -> target capability check
  -> placement
  -> routing
  -> native gate legalization
  -> scheduling/timing resolution
  -> TargetIR verification
  -> executable emission
```

Local execution may emit at QuantumIR; QPUs usually continue to TargetIR.

### 6.3 Static Structure and Late Binding

Evaluate constants, static loops/indices, dead branches, definitions,
target-independent duration expressions, and structural configuration early.
Keep VQE/QML angles, supported sweeps, nonstructural batch parameters, and shots
late. Budget loop/recursion/branch expansion and specialization counts.

### 6.4 Analysis and Invalidation

Bind cached read-only analyses to IR revision, target snapshot, and options.

| Analysis | Consumers | Invalidated by |
| --- | --- | --- |
| DefUseAnalysis | Verifier/DCE/lowering | Values/operands |
| DominanceAnalysis | SSA/control lowering | Blocks/branches |
| QubitLifetimeAnalysis | Ownership/allocation | Alloc/release/control |
| MeasurementDependencyAnalysis | Adaptive legality/batching | Measures/conditions |
| ParameterDependencyAnalysis | Specialization/cache/gradients | Expressions/control |
| InteractionGraphAnalysis | Placement/routing/MPS/TN | Gates/wires |
| CircuitCostAnalysis | Optimization/selection | Gates/layout/duration |
| TargetLegalityAnalysis | Target conversion | Operations/types/snapshot |
| LivenessAndMemoryAnalysis | Plans/buffer reuse | Lifetimes/shapes |
| DistributionAnalysis | Sharded plans | Topology/ownership/communication |

Transformations invalidate all analyses not explicitly preserved; analyses do not
mutate IR. Keys include:

```text
module revision or canonical hash
analysis name and version
analysis options
target/capability hash, when target-dependent
```

Use typed immutable results, not an unversioned global dict. Separate heuristic
costs from correctness facts. Seed randomized routing/placement. Record before/
after hashes, time/peak memory, cache hits, gate/depth/communication/error-cost
changes, preserved/invalidated analyses, diagnostics, reproducible pipelines.

## 7. Compiler/Runtime Boundary

### 7.1 ExecutableArtifact

```text
ExecutableArtifact
  format
  payload or payload reference
  entrypoint
  target identity
  source IR hash
  target IR hash
  compiler version
  pass pipeline digest
  capability snapshot hash
  calibration snapshot hash, when applicable
  parameter schema
  result schema
  approximation/error metadata
```

Formats include runtime plans, openqasm-2/3, qir-base/adaptive, qcis, and native
formats. Preserve protected DeploymentPackage; wrap internally first and migrate
publicly only after a separate proposal.

### 7.2 RuntimeAdapter

```text
submit(artifact, execution_options) -> handle
status(handle) -> status
cancel(handle) -> acknowledgement
result(handle) -> execution result
```

Local synchronous execution may reuse the contract; remote providers have
asynchronous lifecycle. Compiler lowers, optimizes, legalizes, emits. Runtime
selects/authorizes resources, binds parameters/shots, submits, queries/cancels/
retrieves, and records receipts/actual backends. If providers recompile, receipts
separate the submitted artifact from final execution identity; this is not proof
that FlagQuantum emitted pulses.

## 8. TargetCapabilities

Version/hash capability snapshots with:

```text
identity and version
qubit count
native gate set and parameter domains
coupling topology
measurement and reset support
mid-circuit measurement support
adaptive classical operation subset
timing model and feedback limits
accepted artifact formats
OpenQASM/QIR profiles
maximum shots and program limits
parameter binding support
simulator/noise capabilities
```

Separate static capabilities, calibration, account/quota/authorization, and live
availability/queues. Compilation usually depends on the first two; submission
also needs the latter. Account denial does not change program semantics, and
calibration-sensitive scheduling must not reuse stale artifacts.

## 9. OpenQASM 3 Strategy

A 3.0 header is insufficient. Support needs parsing, typing/semantics, scope,
ProgramIR import, target-subset validation, round-trip or explicit asymmetric
conversion, static/dynamic execution tests. Export is only code generation.

```text
Static profile
  gates, parameters, compile-time loops, terminal measurement

Adaptive profile
  mid-circuit measurement, reset, bounded conditionals

Timing profile
  duration, delay, box, stretch, scheduling intent

Pulse profile
  cal, defcal and selected calibration grammar
```

Targets declare exact profiles/features; partial exporters do not imply full
hardware support. Parse on the cold path:

```text
text -> parse once -> IR -> compile/cache -> execute many times
```

Do not parse per shot, interpret each statement/gate in hot Python, return to
cloud Python for coherence-time feedback, reroute every training update, or launch
one dynamic GPU kernel per shot. Static optimization uses folding/unrolling,
merging, layers, fusion, fixed layouts/communication, allocated workspace, late
binding/batches. Dynamic execution uses blocks, shared/batched prefixes,
device-side branch masks, grouped shots, resident state/metadata, explicit
path/memory budgets and reference fallback or rejection. Real-time feedback
belongs on controllers; unsupported targets reject adaptive programs.

## 10. Quantum AI and Distributed Execution

### 10.1 Gradients

Represent parameter identity/binding, observables, requests, exact/approximate
methods, ownership, aggregation. Lower to native autograd, adjoints, shift batches,
or provider gradients. Verify derivatives as well as forward values.

### 10.2 Simulation Representations

```text
QuantumIR
  +-- local statevector plan
  +-- sharded statevector communication plan
  +-- local/rank-owned MPS plan
  +-- tensor-network contraction plan
  +-- JAX kernel plan
```

These are simulation/platform target lowerings, not QPU TargetIR. Preserve
distribution, fallback, and evidence contracts.

## 11. Python and Native Compilation

Start with immutable Python nodes, bounded values/blocks/def-use, verifiers,
PassManager, rewrites, capability models, emitters, internal execution protocols,
differential/property tests. Stable semantics matter before native ABI.

Evaluate C++/MLIR for reproducible compilation bottlenecks, needed dialect
conversion or C++ frontend, LLVM/QIR binary generation/linking, unacceptable large
rewrite costs, native plugin ABI, or C/C++-only SDKs. Restrict native code to a
compiler library; Python retains APIs, training, planning, providers, orchestration.

## 12. API Convergence Boundaries

Permitted preparatory work includes docs/ADRs, characterization, internal opt-in
prototypes, round trips, performance/differential infrastructure, read-only
QASM/provider inventories, compatibility constraints.

Before convergence, do not change public schemas/exports/signatures/behavior,
replace protected packages, stabilize internal IR, rewrite all major paths at
once, or adjust snapshots to hide unapproved changes. Main migration requires
approved Core surface, schema policy, measurement rules, plan/run/compile/
deployment responsibilities, provider-option boundaries, stable tests, and the
convergence branch integrated into mainline.

## 13. Phases

**Phase 0:** design, behavior matrix, serialization/execution characterization,
benchmarks, internal prototypes. Exit: reproducible semantics/limitations/baseline.

**Phase 1:** Python IR/types/values/blocks/diagnostics, verifier/PassManager,
CircuitIR importer, static round trips, state/expectation/gradient parity.
Exit: supported static IR enters without semantic change.

**Phase 2:** canonicalization, decomposition, placement/routing, QASM 2/3/QCIS,
digests/diagnostics/cache, old/new differential tests. Exit: static deployments
meet functional/performance gates.

**Phase 3:** capabilities, TargetIR/legality, artifact/runtime ABI, conformance,
approved DeploymentPackage migration. Exit: local, QASM, and non-QASM targets share
one verified boundary.

**Phase 4:** functions/scopes/blocks/control, measurement def-use, reset/conditionals/
bounded loops, partial evaluation, ProgramIR lowering, QASM Static/Adaptive,
dynamic migration. Exit: shared static/dynamic semantics without batched regression.

**Phase 5:** timing/stretch, QIR Base/Adaptive, scheduling, gradient/distributed
lowering, calibration/pulse references, evidence-based C++/MLIR evaluation.
Each matures independently.

### 13.6 Machine-Verifiable Exits

| Phase | Required checks | Minimum evidence |
| --- | --- | --- |
| 0 | API/IR/execution/deployment/performance baseline | Fixtures, commands, environment, summary, blockers |
| 1 | Import/verify/round-trip/differential | All supported static opcodes, negative fixtures, state/expectation/gradient parity, unchanged API snapshot |
| 2 | Passes/emitters | Golden/property tests, deterministic digest, legal routed gates, QASM/QCIS equivalence, approved budget |
| 3 | Target/artifact/ABI | Three target classes, tamper rejection, submit/status/cancel/result, credential isolation |
| 4 | Program/dynamics | Reset/conditional/reuse/random-branch oracles, QASM profiles, statistical parity, visible performance deviations |
| 5 | Each capability separately | Schema, oracle, target evidence, baseline, maturity; no inherited certification |

```text
compatibility
semantic correctness
determinism and reproducibility
performance and resource bounds
operability and failure behavior
```

Use existing dtype/backend-specific tolerances, not a global architecture error
threshold. Derive performance budgets from Phase 0 measurements. Manifests record:

```text
capability
implementation owner
input/output schema
supported profile
test command and oracle
correctness tolerance
performance budget
public API impact
rollback path
maturity level
evidence artifact
```

Without commands, oracles, and artifacts, entries remain design/experimental.

## 14. Verification

Structural tests cover parser/printer, ordering/hash, unknown fields/ops, schema
migration, source locations. Negative tests cover undefined values, duplicate/
out-of-range/released qubits, measurement dependencies, terminators, unsupported
gates/control/timing, binding phases, forbidden feedback.

Pass tests cover goldens, applicable idempotence, determinism, states,
distributions, expectations, gradients, routed ordering, approximate budgets.
Compare both paths:

```text
legacy path
new IR path
```

Compare state/density, expectations/gradients, samples/counts/order, selected
backend/fallback, distribution, metadata. Provider conformance covers discovery,
format negotiation, lifecycle, bindings, ordering, normalized errors, identity,
credentials, and second-stage compilation receipts.

## 15. Performance Gates

```text
parse_ms
semantic_analysis_ms
lowering_ms
routing_ms
codegen_ms
peak_compiler_memory
artifact_cache_hit_rate
compiled_plan_reuse_count
gate_count_before_after
depth_before_after
kernel_launch_count
host_device_sync_count
shots_per_second
parameter_batches_per_second
dynamic_branch_count
end_to_end_latency
```

Benchmark 1K/10K/100K gates, static loops/subroutines, training, 1/10/100 mid-circuit
measurements, direct IR/QASM 2/QASM 3, SV/MPS/TN and reference/batched dynamics.
Cache hits must avoid repeated parsing/placement/routing/emission. Compiled static
QASM throughput should approach equivalent IR. No new per-gate/per-shot Python
interpreter. Local regressions need budgets/approval; dynamic fallback is visible.
Separate compile, execute, queue, end-to-end timing. Set percentages from measurements.

## 16. Risks

Keep internal IR outside root and public serialization commitments. Implement
only needed bounded compiler features before considering MLIR. Type fields that
affect legality/results/scheduling. Give dual paths exit/removal conditions.
Limit QASM through profiles. Verify gradients/statistics, not gate counts alone.
Allow QuantumIR-to-local-plan emission without mandatory targeting.

## 17. ADR Decisions Before Phase 1

1. Exact public/internal round-trip boundary.
2. Reference-to-linear lowering in Section 5.3.1.
3. Program/request/identity split in Section 5.5.
4. Dynamic integration with ordinary run.
5. Plan/package/artifact relationships.
6. Capability schema maturity.
7. Late binding/cache keys.
8. Provider recompilation receipts.
9. Initial exact QASM 3 profile.
10. Internal persistence/compatibility.
11. Python scale/performance exit thresholds.
12. Evidence required to evaluate C++/MLIR.

## 18. Completion

Require compatible CircuitIR, shared typed static/dynamic semantics, composable/
diagnosable/reproducible passes, pre-submission legality, shared local/QPU
boundaries, replaceable emitters, traceable identities, fast specialization/cache/
batching, forward/measurement/gradient/distributed differential tests, and claims
matching maturity/hardware evidence.

## 19. Candidate API Design

These are post-convergence implementation inputs, not approved contracts.
All public changes need independent proposals, compatibility, tests, owner approval.

### 19.1 Three Surfaces

```text
Stable user surface
  program / plan / compile / run / result

Compiler developer surface
  internal modules / values / operations / passes / analyses

Backend extension surface
  target capabilities / legalization / emitter / runtime adapter
```

Users need no internal IR knowledge; compiler developers do not use root APIs for
values/blocks; providers do not rewrite public IR; Runtime does not reinterpret
source. Credentials/quotas/queues stay out. Internal evolution does not force
Stable Core version changes.

### 19.2 Existing Core

```python
fq.Circuit
fq.CircuitIR
fq.plan(...)
fq.run(...)
```

Keep CircuitIR's name and serialized version, not public CircuitIRV1/V2 classes.
Preserve compiler.compile and Circuit.compile signatures/returns/behavior until
explicit migration approval; no semantic aliases or silent replacement.

### 19.3 Candidate Compilation Entry

```python
from flagquantum.compiler import compile
```

Consider root fq.compile only after stable inputs/outputs, two release cycles of
artifact ABI evidence, clear plan/compile/run responsibilities, complete contracts/
docs/types/errors, and additive proposal approval.

```python
compiled = compile(
    circuit_or_ir,
    target="quafu.baihua",
    options=CompilationOptions(
        optimization_level=2,
        routing_strategy="auto",
        parameter_binding="late",
    ),
)

result = fq.run(
    compiled.artifact,
    options=ExecutionOptions(shots=1000),
)
```

Artifact execution through fq.run is also proposed additive behavior, not an
already granted capability.

### 19.4 Plan, Compile, Run

```text
plan(program, execution requirements)
  -> select representation, target, resources, fallback policy

compile(program, selected target, compilation options)
  -> verified ExecutableArtifact

run(program)
  -> plan + compile + execute

run(artifact)
  -> validate artifact + execute
```

Plans are not execution evidence. Compilation allocates no remote resources,
reads no credentials, submits no jobs. Record requested/selected/actual backends.
Artifact runs do not silently reroute. Receipts record recompilation. Shots/seed/
timeout leave program identity unchanged; targets/pipelines/snapshots change
compilation identity.

### 19.5 CompilationOptions

```python
@dataclass(frozen=True)
class CompilationOptions:
    optimization_level: int = 1
    routing_strategy: str = "auto"
    parameter_binding: str = "late"
    seed: int | None = None
    approximation_budget: float | None = None
    diagnostics: str = "errors"
```

Use closed enums/Literals despite abbreviated sketch types, deterministic
snapshotted defaults, context seeds, no approximation when budget is None.
Target-specific options stay namespaced; credentials/shots/queue/timeout are not
compilation options. Extensions are typed/versioned, not arbitrary mappings.

### 19.6 CompilationResult and Report

```python
@dataclass(frozen=True)
class CompilationResult:
    artifact: ExecutableArtifact
    report: CompilationReport
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class CompilationReport:
    source_ir_hash: str
    target_ir_hash: str
    pipeline_digest: str
    compiler_version: str
    target: TargetIdentity

    gate_count_before: int
    gate_count_after: int
    depth_before: int
    depth_after: int
    routing_swap_count: int

    approximation_error_bound: float | None
    elapsed_ms: float
    pass_statistics: tuple[PassStatistics, ...]
```

Reports are compile evidence, not execution evidence. Define gate/depth per
profile, exclude timing from hashes, distinguish budgets/estimated error, order
diagnostics deterministically, and avoid semantic metadata.

### 19.7 ExecutableArtifact

```python
@dataclass(frozen=True)
class ExecutableArtifact:
    format: str
    payload: bytes
    entrypoint: str

    target: TargetIdentity
    parameter_schema: ParameterSchema
    result_schema: ResultSchema

    source_ir_hash: str
    target_ir_hash: str
    compiler_version: str
    pipeline_digest: str
    capability_snapshot_hash: str
    calibration_snapshot_hash: str | None = None

    @property
    def content_hash(self) -> str: ...
```

```text
flagquantum-runtime-plan
openqasm-2
openqasm-3
qir-base
qir-adaptive
qcis
provider-native:<provider>:<version>
```

Use deterministic bytes or controlled content-addressed references; exclude
credentials/accounts/queues. Validate parameter/result schemas independently.
Keep full target/capability/compiler/pipeline identities; reject tampering/hash/
signature mismatch. Never assume QASM. Large payloads use controlled ArtifactRef,
not arbitrary URLs. Stable serialization needs a separate schema proposal.

### 19.8 Internal Nodes

Use shared nodes and distinct modules, not a universal level-tagged class:

```python
@dataclass(frozen=True)
class Value:
    id: ValueId
    type: IRType


@dataclass(frozen=True)
class Operation:
    name: OpName
    operands: tuple[ValueRef, ...]
    results: tuple[Value, ...]
    attributes: FrozenAttributes
    regions: tuple[Region, ...]
    location: SourceLocation | None


@dataclass(frozen=True)
class Block:
    arguments: tuple[Value, ...]
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class Region:
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Function:
    name: Symbol
    argument_types: tuple[IRType, ...]
    result_types: tuple[IRType, ...]
    body: Region


@dataclass(frozen=True)
class ProgramModule:
    functions: tuple[Function, ...]


@dataclass(frozen=True)
class QuantumModule:
    functions: tuple[QuantumFunction, ...]


@dataclass(frozen=True)
class TargetModule:
    functions: tuple[TargetFunction, ...]
    target: TargetIdentity
    capability_snapshot_hash: str
```

Immutable nodes, deterministic IDs/block/symbol order, source locations excluded
from semantic hashes except explicit debug artifacts. Operation schemas govern
operands/results/attributes/regions. Avoid hard-coded provider isinstance chains.
Extensions register schemas/verifiers/parser/printer/legality. Modules stay internal.

### 19.9 Explicit Conversions

Avoid ambiguous calls:

```python
ir.to_ir(level="target")
ir.lower()
ir.convert("qasm3")
```

Prefer:

```python
import_circuit_ir(circuit_ir) -> QuantumModule

import_openqasm3(source, profile=...) -> ProgramModule

lower_program_to_quantum(
    program,
    context,
) -> QuantumModule

compile_quantum_to_target(
    quantum,
    target,
    context,
) -> TargetModule

emit_executable(
    target_module,
    format,
    context,
) -> ExecutableArtifact
```

Declare levels, profiles, preservation, losslessness, diagnostics, target
requirements, cache keys, round-trip scope. Unsupported reverse conversion returns
structured errors rather than discarding control, timing, measurements, targets.

### 19.10 Pass and Analysis Protocols

```python
class AnalysisPass(Protocol):
    name: str
    version: str

    def run(
        self,
        module: IRModule,
        context: CompilationContext,
    ) -> AnalysisResult: ...


class TransformationPass(Protocol):
    name: str
    version: str
    input_type: type[IRModule]
    output_type: type[IRModule]
    required_analyses: tuple[AnalysisKey, ...]
    preserved_analyses: tuple[AnalysisKey, ...]

    def run(
        self,
        module: IRModule,
        context: CompilationContext,
    ) -> PassResult: ...


@dataclass(frozen=True)
class PassResult:
    module: IRModule
    diagnostics: tuple[Diagnostic, ...]
    statistics: PassStatistics
```

```python
pipeline = PassManager(
    [
        Canonicalize(),
        ConstantFold(),
        DecomposeToBasis(),
        PlaceQubits(),
        RouteQubits(),
        LegalizeTarget(),
    ]
)

result = pipeline.run(module, context)
```

No in-place input mutation. Invalidate unpreserved analyses; analyses are read-only.
Stop on failure; include options/versions in digests; use context seeds. Initially
internal only; external pass plugins need conformance/version negotiation.

### 19.11 Targets and Backends

```python
@dataclass(frozen=True)
class TargetCapabilities:
    identity: TargetIdentity
    n_qubits: int
    native_gates: tuple[GateCapability, ...]
    coupling_map: tuple[tuple[int, int], ...]

    supports_mid_circuit_measurement: bool
    supports_reset: bool
    supports_adaptive_control: bool

    accepted_formats: tuple[str, ...]
    openqasm_profiles: tuple[str, ...]
    qir_profiles: tuple[str, ...]

    schema_version: str

    @property
    def content_hash(self) -> str: ...
```

Compiler backends own no network lifecycle:

```python
class TargetBackend(Protocol):
    def capabilities(self) -> TargetCapabilities: ...

    def legalize(
        self,
        module: QuantumModule,
        context: CompilationContext,
    ) -> TargetModule: ...

    def emit(
        self,
        module: TargetModule,
        context: CompilationContext,
    ) -> ExecutableArtifact: ...
```

Runtime consumes artifacts:

```python
class RuntimeAdapter(Protocol):
    def submit(
        self,
        artifact: ExecutableArtifact,
        options: ExecutionOptions,
    ) -> JobHandle: ...

    def status(self, handle: JobHandle) -> JobStatus: ...
    def cancel(self, handle: JobHandle) -> CancelResult: ...
    def result(self, handle: JobHandle) -> ExecutionResult: ...
```

No credentials in TargetBackend; no program rewriting in RuntimeAdapter. Remote
compilation services return receipts. Separate capabilities, permissions, live
availability. Start in the experimental extension SDK.

### 19.12 Diagnostics and Exceptions

```python
@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal["error", "warning", "remark"]
    message: str
    location: SourceLocation | None
    notes: tuple[DiagnosticNote, ...]
```

```text
FQ-IR-xxxx
FQ-PASS-xxxx
FQ-TARGET-xxxx
FQ-EMIT-xxxx
FQ-ARTIFACT-xxxx
FQ-RUNTIME-xxxx
```

```python
IRValidationError
CompilationError
TargetLegalizationError
ArtifactCompatibilityError
```

Exceptions carry structured diagnostics; do not create public classes per gate/
pass/provider. Warnings/remarks use sinks/reports/log policy, not print.

### 19.13 Layout and Stability

```text
flagquantum/
  compiler/                    # approved public/experimental compiler facade
    __init__.py
    options.py
    result.py
  _compiler/                   # internal implementation
    ir/
    analyses/
    passes/
    pipelines/
    targets/
    emitters/
    diagnostics/
  experimental/
    compiler/                  # early extension surface, if needed
```

| API | Initial status | Stability condition |
| --- | --- | --- |
| fq.CircuitIR | Stable | Existing schema policy |
| fq.plan/run | Stable | Approved convergence contracts |
| flagquantum.compiler.compile | Experimental candidate | Two release cycles, fixed behavior/errors |
| fq.compile | Not present | Separate additive proposal |
| CompilationOptions/Result | Experimental | Fields/defaults/serialization frozen |
| ExecutableArtifact | Internal | ABI/signatures/compatibility/security audit |
| Program/Quantum/TargetModule | Internal | No planned root exports |
| PassManager/AnalysisManager | Internal | External protocol separately designed |
| TargetBackend/Emitter | Experimental extension | Conformance/version/lifecycle |
| Textual internal IR | Debug only | No cross-version promise |

### 19.14 Prohibited Patterns

```python
fq.ProgramIR
fq.QuantumIR
fq.TargetIR
fq.PassManager

CircuitIRV1
CircuitIRV2

ir.to_ir(level="target")
ir.convert("provider-x")

ir.metadata["physical_qubits"]
ir.metadata["gradient_method"]
ir.metadata["runtime_credentials"]
```

Do not replace proposals with snapshot edits, submit paid jobs in compile, silently
rewrite artifacts, leak target options into root, put semantic/legality/identity
fields in Mapping[str, Any], persist debug IR as a long-term contract, publish new
IR class names for internal changes, or equate submission success with execution.

### 19.15 Stabilization Gate

Require journeys/non-goals, exact signatures/types/defaults/errors, serialization/
identity, compatibility/migration, executable semantics, API checking, docs/quick
start/release notes, two different target/runtime conformances, performance and
fallback evidence, owner approval, branch protection/CODEOWNER enforcement.
Approved Stable Core overrides this draft; record differences in ADRs rather than
breaking the approved API to match a sketch.

## 20. References

Repository: AGENTS.md; docs/development/PUBLIC_API_PROTECTION.md;
docs/reference/PUBLIC_API_POLICY.md; docs/roadmap/OPEN_SOURCE_API_QUALITY_PLAN.md;
docs/architecture/RUNTIME_ARCHITECTURE.md; docs/reference/API.md;
capability-maturity.toml.

- [MLIR language](https://mlir.llvm.org/docs/LangRef/)
- [MLIR passes](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [OpenQASM 3](https://openqasm.com/versions/3.0/)
- [QIR specification](https://github.com/qir-alliance/qir-spec)
- [CUDA-Q IRs](https://nvidia.github.io/cuda-quantum/latest/using/extending/compiler/cudaq_ir.html)

References inform design and boundaries; they do not require copying CUDA-Q's
user model or establish current full OpenQASM/QIR/MLIR support.
