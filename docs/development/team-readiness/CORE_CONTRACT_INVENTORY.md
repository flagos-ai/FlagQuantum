# Core contract convergence inventory

Status: in progress

Updated: 2026-09-08

## Conclusion

Core already owns the backend-neutral program representation, artifact envelope,
target-capability vocabulary, numerical requirements, and versioned audit records.
There is no second contract implementation in `runtime/contracts.py`: that file is
the stable Runtime API facade and only re-exports Runtime-owned user types.

The migration is not complete. Stable execution objects still belong to Runtime by
design, while Platform and quantum-cloud provider protocols remain owned by their
provider/deployment implementations. Runtime also owns the concrete evidence
envelope because it collects and verifies live execution facts. These are real
boundaries to converge; moving the Runtime facade into Core would not solve them.

## Current authority map

| Concern | Current authority | Boundary meaning | State |
| --- | --- | --- | --- |
| Circuit semantics | `core/ir.py` | Canonical `CircuitIR`, instructions and measurements | Converged |
| Program artifacts | `core/_artifacts.py` | Internal versioned, content-addressed envelope | Core-owned; limited adoption |
| Target capabilities | `core/target_capabilities.py` | Portable capability facts, requirements and matching vocabulary | Converged |
| Numerical requirements | `core/numerics.py` | Precision, representation and accuracy requirements | Converged |
| Audit contracts | `core/contracts.py` | Requested/estimated/observed execution records and audit decision | Converged |
| Stable execution options | `runtime/options.py` | User intent accepted by `fq.plan` and `fq.run` | Runtime-owned intentionally |
| Stable executable plan | `runtime/execution_plan.py` | Complete executable plan, identity and serialization | Runtime-owned intentionally |
| Stable user result | `runtime/result.py` | Tensor-bearing result and user accessors | Runtime-owned intentionally |
| Runtime API facade | `runtime/contracts.py` | Re-exports Runtime user types; defines no contracts | Not an authority |
| Runtime evidence artifact | `runtime/observability/evidence.py` | Collection, integrity and release-evidence verification | Pending Core envelope boundary |
| Platform provider contract | `providers/platform/contracts.py` | Device lifecycle; currently includes PyTorch types | Pending neutral boundary |
| Execution provider contract | `deployment/cloud.py` | QPU/cloud package, handle, result and provider lifecycle | Pending provider convergence |

## Similar names that are not duplicates

- `ExecutionOptions` is the stable user request. `RequestedExecution` is its small,
  immutable audit projection; it is not accepted as an alternative user request.
- `ExecutionPlan` is executable and carries full identity. `RuntimePlanContract` is
  a lossy audit record produced by `ExecutionPlan.to_contract()`; Runtime must not
  execute it as a substitute for the original plan.
- `ExecutionResult` contains native tensors and user accessors.
  `ExecutionRecordContract` contains portable observed facts for audit.
- `TargetCapabilitySnapshot` describes portable verified target facts.
  Runtime candidate and provider discovery objects are local decisions or probes,
  not competing capability schemas.
- `RuntimeProvenance` records detailed live-run evidence. `ProvenanceContract` is
  the portable audit projection. Collection mechanics do not belong in Core.

These pairs need explicit one-way projection tests, not forced class mergers.

## Remaining convergence work

### 1. Artifact adoption

`ProgramArtifact` is Core-owned and strictly serialized, but the normal Compiler to
Runtime path still passes `CircuitIR` directly. Adopt the envelope only where a
real multi-stage or remote boundary requires persistent identity. Do not wrap every
local function call or replace `CircuitIR` as the canonical program representation.

Completion evidence: one compiled artifact crosses a real execution-provider
boundary with its content identity preserved, without changing the local CPU path.

### 2. Platform provider boundary

`PlatformRuntime` is correctly located with platform implementations today, but its
contract exposes `torch.device` and `torch.Tensor`. A Core-owned platform contract
is justified only when a non-PyTorch platform implementation needs the same
consumer boundary. Until then, keep the working provider-local protocol and do not
create a speculative duplicate.

Completion evidence: two materially different platform implementations satisfy one
neutral conformance suite without consumers importing vendor or framework types.

### 3. Execution provider boundary

`QuantumProvider`, `DeploymentPackage`, `ProviderTaskHandle`, and
`DeploymentResult` remain in `deployment/cloud.py`, while concrete adapters live in
`providers/execution`. The target is a small Core-owned request/result/failure
contract implemented by Simulation and at least one QPU or remote provider. Package
building stays outside Core, and credentials or SDK handles must never enter Core.

Completion evidence: provider replacement requires no Runtime or user-API change,
and both implementations return the same portable result/failure projection.

### 4. Evidence projection

Runtime must continue collecting topology, device residency, communication,
fallback, timing and integrity facts. Core should own only the portable evidence
shape and references. Backend-specific metrics remain namespaced implementation
details rather than fields on a universal evidence dataclass.

Completion evidence: a Runtime evidence artifact projects losslessly into a strict
Core record, and unsupported or silent fallback fails validation.

## Removed debt

- `_compiler` and `compilation` contract authorities have been deleted.
- The unreleased RuntimePlan 0.9 migration hook has been deleted; unsupported
  versions and obsolete field shapes fail closed.
- `runtime/contracts.py` is classified as a facade, so it must not acquire new
  dataclass, protocol, schema or validation definitions.

## Guardrails

- Do not move `ExecutionOptions`, `ExecutionPlan`, `ExecutionResult`, or `Module`
  merely to make their paths look Core-owned; their stable identity and behavior
  remain protected.
- Do not introduce a universal Provider, request, result or evidence object with
  mostly optional fields.
- Do not add a Core contract until an existing type cannot express a current
  cross-domain use case and at least two implementations need the boundary.
- Do not preserve migrations for unreleased schemas.
- Keep cross-domain contract schemas free of framework tensors, vendor SDK,
  network, database and gateway types. Core's existing PyTorch parameter semantics
  are a separate, intentional product choice.

## Verification

The existing Core, API-contract and CPU vertical-slice suites verify strict
serialization, deterministic identity, projection behavior and unchanged user
execution. Architecture and dependency checks prevent Core reverse dependencies.
The migration track remains `in_progress` until the provider and evidence completion
conditions above are demonstrated; documentation alone does not close it.
