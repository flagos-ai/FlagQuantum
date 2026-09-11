# Pre-Release API Quality and Freeze Plan

## 1. Purpose

This plan guides API convergence before FlagQuantum's first public release. It
aims for a small, consistent, extensible, testable public API with sustainable
compatibility, rather than freezing every implementation detail.

It covers strengths, overlap and ambiguity, target namespaces, implementation
order, phase acceptance, and first-release freeze conditions. Mandatory
post-freeze protection, CODEOWNERS, required checks, and authorization are defined
in [Public API Protection](../development/PUBLIC_API_PROTECTION.md).

## 2. Assessment

The main API direction is sound:

```text
Circuit / Module
       v
      IR
       v
     plan
       v
      run
       v
ExecutionResult
```

`fq.Circuit`, FlagQuantum IR, `fq.plan`, `fq.run`, `fq.Module`, `fq.train`, and
shared result models are suitable long-term foundations.

The recorded design assessment is approximately **7.5/10** overall and **8.5/10**
for the main path, with **9/10** as the convergence target. The 60 stable exports
in the original `public_api_v1.json` should not all become permanent first-release
commitments unchanged.

Freeze principles:

1. Stabilize user concepts, core semantics, and compatibility rules, not internals.
2. Keep few stable entries; expose backend-specific features through namespaces or experimental APIs.
3. Keep simple paths short and expert paths inspectable, reproducible, and controllable.
4. Execute the plan shown to the user.
5. Prefer compatible additions; removals require formal deprecation.

## 3. Strengths to Retain

### 3.1 Shared Program Model

`fq.Circuit` uses versioned IR with deterministic serialization, content hashing,
parameter expressions, measurements, and verifiable errors. This supports a
backend-independent ecosystem.

### 3.2 Separate Responsibilities

`fq.plan` plans and explains; `fq.run` executes once; `fq.Module` owns PyTorch
parameters; `fq.train` runs an optimizer loop; `ExecutionResult` and
`TrainingResult` provide stable output. Preserve this division.

### 3.3 Native PyTorch Experience

`fq.Module` is a regular `torch.nn.Module` supporting parameter management,
optimizers, device movement, and autograd. Keep PyTorch as the main interface
rather than inventing another training-object system.

### 3.4 Governance Foundations

The repository already has stable snapshots, executable contracts, documentation
source-of-truth, experimental namespaces, DeprecationWarning paths, semantic
versioning, and release policy. Consolidate these foundations.

## 4. Required Pre-Release Issues

### API-001: ExecutionPlan Must Be Executable — P0

The original `fq.plan(circuit)` path mainly inspected/displayed plans while
`fq.run(circuit)` replanned. The target is:

```python
result = fq.run(circuit, options=options)

# Equivalent inspectable path.
plan = fq.plan(circuit, options=options)
result = fq.run(plan)
```

`fq.run(ExecutionPlan)` must execute that plan, which carries program/config
hashes and environment constraints. Stale plans fail explicitly without silent
replanning. Automatic and explicit paths have equivalent plans/results, and
`ExecutionResult.plan` identifies the actual plan.

### API-002: Competing Planning Entries — P0

`fq.plan`, `Circuit.plan`, `Circuit.runtime_plan`, and `plan_runtime_selection`
answer overlapping but different questions. Recommend only `fq.plan` in user
documentation. `Circuit.plan()` may remain only as a strict equivalent shortcut.
Runtime selection belongs inside `ExecutionPlan`; specialist analyzers belong
in `fq.planning` or internal namespaces, not as permanent root planners.

### API-003: Competing Execution Entries — P0

The original surface contains `fq.run`, `Circuit.run`, `fq.run_native`,
`fq.run_mps`, `fq.run_tensor_network`, `fq.run_target`, and specialized distributed/
noisy runners. Recommend one entry:

```python
fq.run(program_or_plan, options=...)
```

`Circuit.run()` must be equivalent. Backend-native execution belongs in
`flagquantum.runtime`; numerical expert entries belong in Simulation method
packages. Unstable distributed/hardware/research paths belong in `fq.experimental`.
Specialist root runners are not long-term Stable Core.

### API-004: Multiple Configuration Authorities — P0

Circuit construction, `RuntimeConfig`, `RuntimePolicy`, run keyword options,
environment variables, and backend parameters can all influence execution. This
obscures precedence and reproducibility. Use immutable shared options:

```python
options = fq.ExecutionOptions(
    mode="auto",
    backend="auto",
    device="auto",
    batch_size=1,
    precision="complex64",
)
```

Specify precedence explicitly; the proposal illustrates:

```text
Explicit ExecutionOptions
 > Module RuntimePolicy
 > Defaults captured at Circuit construction
 > Process defaults
 > Framework defaults
```

Reject unknown options rather than silently accepting arbitrary `**options`.

### API-005: mode/backend/device/target Overlap — P0

| Concept | Meaning | Examples |
| --- | --- | --- |
| `mode` | Mathematical representation | statevector, mps, tensor_network, density_matrix |
| `backend` | Execution implementation | pytorch, jax, triton, vendor |
| `device` | Execution device | cpu, cuda, npu, flagos |
| `target` | Requested output | state, expectation, samples, amplitudes |

Use identical names, enums, and defaults in plan, run, and RuntimePolicy. Aliases
belong only in compatibility layers with removal versions.

### API-006: Naming Inconsistency — P0

Resolve `run(mode=...)` versus `plan(state_mode=...)`, `bsz` versus `batch_size`,
and `n_qubits`/`n_wires`/`nqubits`. The proposal recommends public `mode`,
`batch_size`, and Circuit `n_qubits`; reserve wire terminology for IR/mapping/
internal indices. Migrate private callers before making historical spellings a
public compatibility commitment.

### API-007: Two Measurement Sources — P0

Proposal 004 chose one canonical model: requests end in
`ExecutionPlan`'s `CircuitIR.measurements`. `measurements=` is a planning shortcut
only; reject it when IR already has measurements rather than override or append.
Implemented rules use IR by default, insert explicit requests into empty IR before
planning, and reject simultaneous sources.

### API-008: Result Boundaries — P0

Proposal 004 narrowed program types in run/plan, removed `Any` from
`ExecutionResult.plan` and `MeasurementResult.value`, added fail-closed required
accessors while retaining optional composition fields, and removed native
`__getattr__` forwarding. Proposal 005 added a versioned diagnostics envelope for
`metrics/runtime/provenance/compatibility`; internal section keys may grow compatibly.

Target returns:

```python
fq.run(...) -> fq.ExecutionResult
fq.plan(...) -> fq.ExecutionPlan
fq.train(...) -> fq.TrainingResult
```

Suggested explicit accessors:

```python
result.expectation()
result.statevector()
result.require_samples()
result.measurement("energy")
result.native()  # Explicitly outside the stable backend-neutral contract.
```

Missing results raise clear errors, stable fields have checkable types, native
objects do not expand the API implicitly, metadata is versioned, and serialization
or nonserializable fields are explicit.

### API-009: Root Surface Too Large — P0

The original 60 exports mix MPS production/evidence types, native/distributed
runners, deployment helpers, and information conveniences. Their maturity differs.
Use three levels.

**Stable Core:** approximately 15–25 candidates:

```text
Circuit, CircuitIR, Instruction
Parameter, ParameterExpression
MeasurementNode, MeasurementResult, ObservableNode
Module, RuntimePolicy
ExecutionOptions, ExecutionPlan, ExecutionResult
TrainingOptions, TrainingResult
plan, run, train
experimental
__version__
```

**Stable Extensions:** namespaced capabilities:

```text
fq.noise
fq.deployment
fq.interop
flagquantum.runtime
flagquantum.simulation
fq.compiler
```

**Provisional/Experimental:** ranks/shards/transports, MPS production gates and
benchmark evidence, hardware controls, incomplete dynamic circuits/backends, and
research planners/policies.

### API-010: Loose Compatibility Root — P0

Lazy `__getattr__`/`__dir__` can expose historical entries outside stable
`__all__`. Users reasonably interpret root accessibility as public API.
Discover only Stable Core and explicit namespaces; migrate/remove old entries
before release, put necessary compatibility in `flagquantum.compat`, and avoid
native attribute forwarding.

## 5. Additional Pre-Release Priorities

### API-011: compile/plan/run Responsibilities — P1

Clarify `Circuit/IR -> compile -> plan -> run`, or make compilation a planning
stage. Users should not manually combine redundant work. Recommend automatic
normalization, compilation, routing, and resources in `fq.plan`; expert
`fq.compiler.optimize` for target-independent optimization and
`fq.compiler.compile` for targets. Plans retain compiled IR and transformation
provenance with stable identity for identical inputs/options.

### API-012: Module.forward/execute/run Relationship — P1

`Module.forward()` returns Tensor for PyTorch; `Module.execute()` returns
`ExecutionResult`; `fq.run()` primarily accepts Circuit/IR. Fix gradient, batch,
and observable defaults across entries.

**Proposal 005: implemented, pending separate freeze approval.** `forward`
returns an autograd Tensor, `execute` returns a result with plan/provenance,
`fq.run` rejects Module, and Module has no additional `run`. Both Module paths
share owned/override parameters, observables, and backend policy. No implicit
input-batch × parameter-batch Cartesian broadcasting exists.

### API-013: Module Construction Variants — P1

Validate builder, parameterized Circuit, flat/named parameters, and initialization
through golden paths. Keep a simple builder and named Parameter path, specify
batching, place complex bindings in configuration/classmethods, and test every
public construction form.

Proposal 005 protects existing forms without adding top-level construction modes.
Proposal 006 removed unvalidated deployment binding from Module; applications or
Deployment own it. Old checkpoint extra-state still reads and ignores the field.

### API-014: Training Lifecycle Scope — P1

The original `fq.train` covers optimizer, objective, steps, inputs, logging, and
callbacks, not a unified checkpoint/resume/early-stop/validation/distributed
lifecycle. Choose either a complete TrainingOptions/callback protocol or an
explicit minimal optimizer loop.

**Proposal 005 chose the minimal loop.** Checkpoint/resume belongs to
`Module.save_checkpoint/load_checkpoint` or application loops. Early stopping,
validation sets, and implicit recovery do not enter the stable top-level signature.
Documentation and claims must match.

### API-015: Distributed Training Split — P1

`train_distributed_statevector` and `train_distributed_mps` expose implementations.
The longer-term candidate is:

```python
fq.train(module, options=fq.TrainingOptions(execution=...))
```

**Proposal 005:** distributed training remains in
`flagquantum.experimental.distributed` until result, checkpoint, gradient, and
optimizer semantics match local Module. It does not enter stable root prematurely.

### API-016: Parallel Noise Entries — P1

Existing `noise_model=`, density, noisy MPS, and noisy statevector entries overlap.
Recommend ordinary `fq.run(..., noise=...)`, mode selection through options/planning,
expert simulators in their Simulation packages, shared results, and planning-time
failure for unsupported noise capabilities.

### API-017: Error Model — P1

Use a small hierarchy while retaining underlying exceptions as `__cause__`:

```text
FlagQuantumError
├── ValidationError
├── PlanningError
├── CompilationError
├── ExecutionError
├── CapabilityError
└── SerializationError
```

Users should not depend on incidental PyTorch/JAX/NCCL/provider error text.

**Proposal 006: implemented, pending separate freeze approval.** Seven categories
live in stable `flagquantum.errors`. Multiple inheritance preserves built-in
compatibility. Wrong types retain `TypeError`; values, planning, serialization,
capability, compilation, and execution/training use explicit categories. Existing
IR, Plan, and TrainingState error names remain.

### API-018: Plan Portability and Serialization — P1

Separate local executable plans from deployable packages:

```text
ExecutionPlan       Inspectable/executable locally, with environment constraints
DeploymentPackage  Serializable/signable, submitted to hardware/providers
```

Do not overload one ambiguous object with both responsibilities.

### API-019: Stable Extension Protocols — P1

Third parties should add backends/providers/passes/measurements without modifying
Core. Stabilize protocols and registration rather than concrete implementations:
capabilities, planner costs/contributions, runners, result normalization,
submission, version negotiation, and conformance.

## 6. Target API Sketch

These sketches are targets, not permanent freeze declarations before implementation.

```python
import flagquantum as fq

circuit = (
    fq.Circuit(n_qubits=2)
    .h(0)
    .cx(0, 1)
)
options = fq.ExecutionOptions(
    mode="auto",
    backend="auto",
    device="auto",
    batch_size=1,
)

# Short path.
result = fq.run(circuit, options=options)

# Inspectable, reproducible path.
plan = fq.plan(circuit, options=options)
print(plan.summary())
result = fq.run(plan)
```

Training:

```python
module = fq.Module(build_circuit, parameters={"theta": ()})
training = fq.train(
    module,
    optimizer=optimizer,
    objective=objective,
    options=fq.TrainingOptions(steps=100),
)
```

Expert native execution:

```python
from flagquantum.simulation.mps import run_mps
native = run_mps(circuit, options=options)
```

Unified execution remains:

```python
result = fq.run(circuit, options=options)
```

## 7. Five Golden User Paths

Before freeze, execute real examples for:

1. Bell construction, execution, state/measurement reads.
2. Parameterized VQE with Parameter, Hamiltonian, gradients, optimization.
3. `fq.Module` inside ordinary `nn.Module` and optimizers.
4. Planning the same code across statevector, MPS, TN.
5. Qiskit conversion, deployment packages, hardware/provider submission.

Each imports only `flagquantum as fq` or stable namespaces, avoids runtime/
internal/testing/evidence internals, runs in CI, supplies stable actionable errors,
at least builds/plans on CPU, and links plan/execution/result provenance.

## 8. Implementation Phases

### Phase 0: Baseline and User Journeys

Preserve `public_api_v1.json` as the internal baseline. Record exports, signatures,
defaults, and dataclass fields in `contracts/public-api-v0.2-baseline.json`, enforced
by `tools/public_api_snapshot.py`. Add golden paths, inventory signatures/docs,
and stop root API expansion during convergence. Acceptance: automatically detect
public signature/example changes.

Implementation record (2026-08-31):

- Baseline records 60 exports; snapshot checks run in pre-commit and `CI / quality`.
- `tests/api_contract/test_open_source_golden_paths.py` covers Bell, parameterized
  objective training, PyTorch composition, planning, local deployment, Qiskit round trips.
- Core CI runs the first five paths; optional Qiskit 2.0/2.5 matrix runs Qiskit.
- This is a migration baseline, not approval of all 60 final exports.

### Phase 1: Reduce Stable Surface

Choose Stable Core, relocate backend/evidence/acceptance types, remove unnecessary
root compatibility, update manifests/docs/import contracts. Target approximately
15–25 exports, each with a user scenario.

Implementation record (2026-08-31):

- `contracts/public-api-v1-candidate.json` classifies each of 60 exports exactly once.
- Candidate Core has 22 entries: 20 retained plus `ExecutionOptions`/`ExecutionPlan`.
- Remaining entries map to stable extensions, experimental, or pre-release removal.
- `API_CHANGE_PROPOSAL_001_STABLE_CORE.md` records migration; API owner approved
  classification/namespace migration on 2026-08-31. Final freeze remains separate.
- Stable `backends`, `compiler`, `deployment`, `noise`, `operators` and experimental
  `distributed`, `mps`, `planning` namespaces import successfully.
- Golden deployment examples no longer depend on old root helpers.
- README, architecture, maintained guides/reference, and examples migrated;
  `tools/check_legacy_root_api_usage.py` prevents regression in pre-commit/CI.
- Production, tools, and benchmarks migrated. `benchmarks/mps_stability.py` remains
  an exact read-only historical exception because its source hash binds measured
  A800 evidence; do not update the hash without remeasurement just for imports.
  Remaining work at that checkpoint was tests and generated audit docs.
- Test batches removed 21 calls in 16 files, then 58 in 13 files, 113 in 2 files,
  and 152 in the last 6 files covering MPS/native/trajectory/JAX/planners.
  `contracts/legacy-root-api-test-debt.json` is now a zero baseline enforced by CI.
- `fq.__all__`, `dir(fq)`, and `docs/public_api_v1.json` shrank from 60 to 22.
  Proposal 002 approved ExecutionOptions; Proposal 003 implemented/approved
  ExecutionPlan root listing. Overall freeze still awaited semantic proposals.
  Historical lazy access was temporarily uncommitted compatibility, outside the
  manifest, with a separate removal inventory.
- Proposal 001 migrated/removed entries are closed in root `__getattr__` with
  replacement namespace messages. Historical `flagquantum.api` aggregation and
  implicit root fallback were removed before public alpha; callers use domain authorities.

### Phase 2: Names and Options

Introduce ExecutionOptions, unify mode/backend/device/target, replace state_mode
and bsz, expose Circuit n_qubits, fix precedence. Acceptance: no conflicting terms
in plan/run/Circuit/RuntimePolicy.

Implementation record (2026-08-31):

- `API_CHANGE_PROPOSAL_002_EXECUTION_OPTIONS.md` and
  `contracts/execution-options-v1-candidate.json` registered.
- Implementation, root authorization, default/runtime/distributed validation complete.
- Shared stable input uses immutable field overlays, strict precedence, no `extras`
  escape hatch, and approximation/backend fallback prohibited by default.

### Phase 3: Executable Plans

Design record (2026-08-31): Proposal 003 and
`contracts/execution-plan-v1-candidate.json` registered. Identity, serialization,
stale plans, environment constraints, and `fq.run(plan)` are implemented and
validated on default/runtime/distributed selections. Root export approved; final
freeze separate. ExecutionPlan is locally inspectable/cacheable/restorable/
executable; DeploymentPackage owns signing, submission, remote lifecycle.

Return stable plans, execute them directly, fingerprint program/options/environment,
report stale plans, eliminate duplicate compilation/silent replanning. Acceptance:
automatic and explicit paths have equivalent output/provenance.

### Phase 4: Results and Measurements

- [x] Remove `Any` from stable program, plan, and measurement values.
- [x] Define measurement sources/conflicts.
- [x] Add stable result accessors.
- [x] Isolate backend-native objects.
- [x] Version result summaries.
- [x] Add Proposal 005 Module diagnostics envelopes.
- [ ] Unify public missing-data and training errors in a later proposal.

Acceptance: users read requested results without probing ambiguous optional fields.

### Phase 5: Module and Training

- [x] Fix forward/execute/run relationships.
- [x] Fix owned/override parameters, inputs, observables; reject implicit Cartesian batching.
- [x] Assign checkpoint/resume to Module or application loops.
- [x] Keep distributed training experimental.
- [x] Fix PyTorch primary/JAX optional compiled-backend boundaries.
- [x] API owner separately approves Proposal 005 freeze.

Proposal 006 removes provider/deployment state and fixes three construction paths.
Module owns parameters, RuntimePolicy, precision, training state. Eager, train,
and explicit execution must share defaults.

### Phase 6: Extensions and Interoperability

- [x] Define backend/provider Protocol candidates.
- [x] Establish third-party conformance.
- [x] Validate Qiskit/PennyLane support-window endpoints in real-dependency Docker and matching CI.
- [x] Verify plugins need no Runtime internals.
- [x] API owner separately approves Proposal 007 freeze.

Acceptance: add a minimal third-party backend without modifying Core.

### Phase 6A: Public Errors

- [x] Establish stable `flagquantum.errors`.
- [x] Preserve ValueError/RuntimeError/NotImplementedError catch compatibility.
- [x] Map IR/Plan/Result/Module/training-state errors to shared categories.
- [x] Isolate stable boundaries from incidental native exceptions.
- [x] API owner separately approves Proposal 006 freeze.

### Phase 7: Candidate and Freeze

Release alpha; exercise real examples, internal applications, and at least one
external integration. In beta stop arbitrary naming changes and fix contracts.
Generate final manifests, typing snapshots, migration notes, and release audit.

## 9. Quality Gates

**Surface:** root matches manifest; new exports reviewed; experimental excluded;
examples avoid internals.

**Signatures:** positional additions reviewed; stable returns not `Any`; options
checkable; defaults snapshotted.

**Semantics:** automatic/explicit plans equivalent; stale plans and unsupported
capabilities fail closed; no measurement override; fallback visible in results.

**Documentation:** first README example and five golden paths run in CI; API docs
generated from manifest; each stable export has one main document and executable contract.

**Compatibility:** removals have deprecation/removal versions and replacement
warnings; retain compatibility across at least one public minor release; IR,
Plan, Result, DeploymentPackage each have schema versions.

## 10. First-Release Freeze Conditions

All must hold:

- [x] Stable Core reduced and individually reviewed.
- [x] ExecutionOptions is the sole recommended execution configuration.
- [x] `fq.run(plan)` implemented and equivalent.
- [x] plan/run/RuntimePolicy terminology aligned.
- [x] Measurement source/override rules unambiguous.
- [x] run/plan/train return types are not `Any`.
- [x] Native objects do not implicitly expand stable Result.
- [x] Module forward/execute/train candidates implemented and contract-tested.
- [x] Checkpoint/resume ownership fixed.
- [x] Module construction/deployment-binding ownership fixed.
- [x] Public error candidate hierarchy implemented and tested.
- [x] Distributed/noise/backend specialist entries layered.
- [x] No accidental compatibility root exports.
- [x] Five golden paths pass.
- [x] Qiskit/PennyLane conformance passes.
- [x] Third-party backend example uses public extension protocols only.
- [x] Documentation, typing, snapshots, release notes agree.
- [ ] Alpha/beta use reveals no issue requiring an API break.

## 11. Version Strategy

```text
Private phase: concentrated breaking convergence permitted
Public alpha: documented API candidate adjustments permitted
Public beta: stop arbitrary renaming; fix contract defects
First stable release: compatibility commitment; prefer additive evolution
```

Versions below 1.0 may migrate explicitly under semantic versioning, but a low
version is not justification for frequent user breakage. Every public breaking
change needs concrete benefit, migration, and a deprecation window.

## 12. Recommended Order

Reduce root exports; unify options/terminology; implement executable plans;
tighten results/measurements; converge Module/training; establish extensions;
validate alpha/beta journeys; freeze last.

The resulting contract should let users learn few core objects, experts inspect
and pin plans, and third parties extend protocols while internal backends,
compilers, and hardware adapters continue to evolve.
