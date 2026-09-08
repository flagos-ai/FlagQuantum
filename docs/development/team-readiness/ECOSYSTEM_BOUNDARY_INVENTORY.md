# Ecosystem Boundary Inventory

Status: team readiness inventory

Owner: Ecosystem

Worktree: `FlagQuantum-vNext-ecosystem`

Branch: `codex/vnext-team-ecosystem`

Inventory date: 2026-09-03

## Decision

FlagQuantum IR remains the only canonical program representation. Qiskit
`QuantumCircuit`, PennyLane `QuantumScript`, provider SDK handles, JAX arrays,
and future CUDA-Q/QX objects may exist only inside their owning adapters or
kernel/provider implementations. An import adapter must return `CircuitIR` plus
FlagQuantum-owned diagnostics before Compiler, Runtime, or Simulation sees the
program. An export adapter may return an external artifact only in the explicit
interop result's `artifact` field.

PyTorch is a deliberate exception to the phrase “external objects only at the
boundary”: it is the primary public training and numerical interface. Stable
`fq.Module`, parameter binding, and execution results intentionally use
`torch.Tensor`. This does not authorize Qiskit, PennyLane, CUDA-Q, provider, or
JAX objects in owned contracts. JAX is an optional quantum-kernel accelerator;
its arrays must be converted across the DLPack/autograd boundary and must not be
returned as canonical IR or public execution results.

This inventory changes no Stable Core surface and performs no large-scale file
move. The inspected execution paths are `single_device_fast_path` ecosystem
conversion and local conformance paths; no distributed scalability claim is
made.

## Boundary model

| Concern | Accepted input at outer edge | Owned value passed inward | External value allowed on return | Owning layer |
| --- | --- | --- | --- | --- |
| Format interoperability | Qiskit circuit, PennyLane script, OpenQASM/QCIS text | `CircuitIR`, compiler target IR, typed conversion report | Explicit exported circuit/script/text artifact | Ecosystem for framework conversion; Compiler for target text emission |
| Machine-learning frontend | PyTorch tensors/modules; future framework-native frontend object | `CircuitIR`, named parameters, `RuntimePolicy` | PyTorch tensor/result by stable design | PyTorch Runtime; Ecosystem only for non-native frontend translation |
| Optional quantum kernel | PyTorch tensor at autograd bridge | Kernel-local JAX arrays | PyTorch tensor and gradient | Runtime optional JAX backend |
| Plugin extension | Extension implementation and manifest | Capability request/response and owned configuration | Extension result behind negotiated handle | `ecosystem/extensions/**` |
| Backend execution | Owned program/plan or deployment package | Provider-neutral execution/deployment contract | Owned execution result plus serializable provenance | Runtime or Execution Provider, never format interop |
| Algorithm example/benchmark | Library-specific demo object | No production contract | Benchmark-only payload with explicit methodology | `examples/**` or `benchmarks/**` |

## Entry-point inventory

### Qiskit

| Operation | Entry point | Boundary result | Classification | Finding |
| --- | --- | --- | --- | --- |
| Discover/register | `flagquantum.ecosystem.available_adapters`, `get_adapter("qiskit")`, `QISKIT_ADAPTER` | Lazy `InteropAdapter`; no Qiskit import during discovery | Format interoperability | Correct boundary |
| Import | `import_qiskit`, `from_qiskit` | `QiskitImportResult(ir=CircuitIR, report=...)` or `CircuitIR` | Format interoperability | Correct object boundary; metadata caveat below |
| Export | `export_qiskit`, `to_qiskit` | Qiskit object only in explicit `artifact`/`circuit` | Format interoperability | Correct boundary |
| Semantic conversion | `qiskit_statevector_to_flagquantum`, `semantic_fingerprint`, `run_qiskit_conformance` | PyTorch tensor or owned conformance record | Result conversion/test | Correct boundary |
| Execute | `run_qiskit_aer_dynamic`, `run_qiskit_aer_qasm3_round_trip` in `interop/qiskit/execution.py` | `DynamicExecutionResult` | Backend execution | Misclassified location; migrate after an execution-provider contract exists |

Import behavior is fail-closed by default. Barriers, arbitrary metadata,
register flattening, unsupported control flow, unsafe multi-qubit unitary basis
order, unsupported parameter expressions, observables, and measurement requests
are represented by typed issue codes. `allow_lossy=True` is the only opt-in to a
partial conversion.

### PennyLane

| Operation | Entry point | Boundary result | Classification | Finding |
| --- | --- | --- | --- | --- |
| Discover/register | `get_adapter("pennylane")`, `PENNYLANE_ADAPTER` | Lazy `InteropAdapter`; no PennyLane import during discovery | ML frontend/format interoperability | Correct boundary |
| Import | `import_pennylane`, `from_pennylane` | `PennyLaneImportResult(ir=CircuitIR, report=...)` or `CircuitIR` | Static format interoperability | Correct boundary |
| Export | `export_pennylane`, `to_pennylane` | `QuantumScript` only in explicit export artifact | Static format interoperability | Correct boundary |
| Conformance | `run_pennylane_conformance` | Owned complex128 conformance record | Result conversion/test | Correct boundary |

The current v1 contract is intentionally static and IR-only. It does not admit
PennyLane runtime execution, PennyLane autograd objects, trainable PennyLane
parameters, shots, or measurement processes into IR. Non-contiguous/string wire
labels require explicit lossy flattening. Idle wire extent cannot be preserved
by a bare `QuantumScript` and therefore fails closed.

### PyTorch and JAX

| Framework | Entrypoints | Classification | Boundary assessment |
| --- | --- | --- | --- |
| PyTorch | `fq.Module`, `fq.Circuit` tensor parameters/matrices, `fq.run`, `ExecutionResult`, extension backend conformance | Primary ML frontend and numerical runtime | Intentional owned product surface; not an ecosystem adapter. Other framework objects must not piggyback through tensor-valued fields or metadata. |
| JAX | `runtime/backends/jax/**`, DLPack helpers in `kernel.py`, PyTorch autograd wrapper | Optional backend execution/kernel acceleration | Imports are confined to the optional JAX backend. JAX arrays are kernel-local and outputs return through PyTorch. |

The v0.1 device-coupled encoding API has been removed. Parameterized
`fq.Circuit`/`fq.Module` construction is the maintained PyTorch frontend.

### Formats and provider SDKs

| Format/provider | Current entry points | Classification | Boundary assessment |
| --- | --- | --- | --- |
| FlagQuantum IR JSON/dict | `CircuitIR.to_dict/from_dict`, compiler `import_circuit_ir`/`export_circuit_ir` | Canonical owned format | Source of truth |
| Legacy engine “QIR” | `Circuit.to_qir/from_qir`, `core.from_engine_qir` | Compatibility format | This is a legacy Python instruction-list format, not LLVM/Microsoft QIR; rename/deprecate only through Stable Core process |
| OpenQASM 2 static | `compiler/openqasm.py`, offline deployment | Target format export | Compiler owns the only static OpenQASM lowering path |
| OpenQASM 3 static | compiler artifact profiles/emitter/offline deployment | Target format export | Compiler-owned target lowering |
| OpenQASM 3 dynamic | `runtime/dynamic/dialects/openqasm3.py` | Runtime dynamic dialect export | Format lowering is embedded in Runtime and should move after a shared dynamic artifact contract is approved |
| Braket IQM dynamic QASM | `runtime/dynamic/dialects/braket_iqm.py` | Vendor backend dialect | Vendor vocabulary in Runtime; move to Execution Provider boundary |
| QCIS v1 | `compiler/qcis.py` | Target format export | Compiler owns the only QCIS emitter |
| Amazon Braket SDK | `providers/execution/braket.py` | Provider discovery/submission | Correct provider boundary; SDK objects should never enter IR/runtime records |
| Quafu/QuarkCircuit | `deployment/providers.py` and calibration helpers | Provider discovery/submission | Correct provider boundary; keep device/job objects local |

`architecture.toml` already confines direct Qiskit and PennyLane imports to
their interop namespaces. The new team tests additionally scan Core, Compiler,
Runtime, and Simulation for external SDK imports and confine JAX imports to the
optional JAX backend.

### Extension SDK and registry

`flagquantum.ecosystem.extensions` exposes manifests, capability negotiation, scoped
registration, lifecycle containment, and conformance helpers. There is one
registry (`ExtensionRegistry` with a task-local `ContextVar`), so no second
plugin registry should be introduced.

The protocols currently use broad `Any` values for program, parameter, result,
compiler transform, and planner hooks. This is a deliberate transitional risk:
an extension could pass a vendor object through a handle even though the built-in
framework adapters do not. Tightening these methods requires an Ecosystem and
integration contract change using FlagQuantum-owned program, plan, result, and
serialized provider payload types.

### Algorithms, examples, and benchmarks

Production algorithms under `flagquantum/algorithms/**` use PyTorch and
FlagQuantum APIs, not Qiskit/PennyLane/CUDA-Q objects. External comparison code
belongs in examples or benchmarks. The existing CUDA-Q comparison and gradient
capability probes are benchmark-only and do not establish a CUDA-Q production
adapter or alternate IR.

## Object-leakage audit

### Confirmed clean boundaries

- No direct Qiskit, Qiskit Aer, PennyLane, Cirq, CUDA-Q, Braket, or Quark SDK
  import was found in Core, Compiler, Runtime, or Simulation.
- Qiskit and PennyLane imports are lazy and occur only when an adapter action is
  explicitly requested.
- Normal Qiskit/PennyLane import results contain FlagQuantum dataclasses,
  primitives, mappings, tuples, and allowed PyTorch tensors; the external
  circuit/script stays in the explicit export artifact.
- JAX imports are confined to `flagquantum/runtime/backends/jax/**`; the public
  training/result side remains PyTorch.

### Leakage and coupling register

| Severity | Location | Finding | Why it matters | Required owner/action |
| --- | --- | --- | --- | --- |
| P1 | `interop/qiskit/conversion.py` source provenance copy | Recognized `flagquantum_*` metadata values are copied recursively without an owned scalar/container validator. A caller can place an arbitrary external object under one of those keys and carry it into `CircuitIR.metadata`. | Potential real object leakage through an otherwise correct adapter. | Core must define canonical metadata value types and validation; Ecosystem then rejects or explicitly serializes unsupported values with an approved issue code. |
| P1 | `ecosystem/extensions/sdk.py` protocols | `execute`, `value_and_grad`, `transform`, and `plan` accept/return `Any`. | Third-party objects can cross layers through a negotiated extension. | Ecosystem/integration must replace `Any` at cross-layer points with approved owned contracts. |
| P2 | `runtime/dynamic/conformance.py` | Runtime compatibility wrappers import `interop.qiskit.execution`. | Dependency direction is Runtime -> Ecosystem. | Move Qiskit Aer implementation to Execution Provider; keep an Ecosystem format converter and a compatibility shim with an owned removal plan. |
| P2 | `_compiler/importers/circuit_ir.py` | Compiler provenance allowlist names `qiskit_label`. | No external object leaks, but vendor vocabulary has entered Compiler. | Core defines vendor-neutral operation label/provenance semantics; Qiskit maps at the edge. |
| P2 | `runtime/dynamic/dialects/braket_iqm.py` | Vendor-specific Braket/IQM lowering is implemented under Runtime. | Runtime owns orchestration, not vendor artifact dialects. | Execution Provider migration after a dynamic artifact contract is approved. |
| P3 | `encoding/encoder.py` | Legacy PyTorch/device frontend directly invokes old device operations. | It bypasses the modern IR-centered user journey but does not import another ecosystem. | Migrate examples/users to `fq.Circuit`/`fq.Module`; retire only through compatibility policy. |

The Runtime-to-Ecosystem reverse dependency recorded above has since been
removed: dynamic conformance tests now invoke the Qiskit adapter at its owning
Ecosystem boundary. The P1 metadata issue is intentionally documented rather
than patched here:
adding a canonical metadata value algebra or a new public issue code touches
protected Core/interop contracts and must be approved contract-first. The team
tests cover normal-result object containment and prevent new direct SDK import
leaks; they do not falsely certify arbitrary metadata as safe.

## CUDA-Q and QX migration placement

No production CUDA-Q or QX adapter exists today. “QX” is not defined in the
current repository, so its exact source artifact and runtime must be specified
before implementation. Regardless of vendor naming, migration follows the same
split:

1. Put program/object import and export in
   `flagquantum/ecosystem/<framework>/`. Convert immediately to/from `CircuitIR`;
   never store the vendor kernel, builder, AST, MLIR/QIR handle, observable, or
   result object in FlagQuantum IR.
2. Put execution, target discovery, job submission, polling, cancellation, and
   provider result decoding behind the approved Execution Provider/deployment
   contracts. A format adapter must not become a hidden backend.
3. Put a genuine optional numerical kernel under
   `runtime/backends/<kernel>/` only if it consumes owned lowered data and
   returns owned PyTorch-facing results. Kernel-local vendor objects must not
   appear in plans, checkpoints, or serialized evidence.
4. Keep comparison scripts in `benchmarks/**`. They are evidence consumers, not
   registries or production implementations.
5. Do not import CUDA-Q MLIR, QIR, QX bytecode, or vendor circuit classes as a
   second canonical program model. Any multi-level compiler representation must
   be FlagQuantum-owned and derived from canonical FlagQuantum IR.

## Target directory migration order

This is sequencing only; no bulk move is part of this change.

1. **Contract prerequisite.** Core/integration defines canonical metadata
   value types, neutral operation labels/provenance, wire/bit semantics, and the
   dynamic artifact/execution-provider handoff.
2. **Guard the current boundary.** Land the team tests for direct SDK imports,
   JAX locality, round trips, semantic state equivalence, failure issue codes,
   and external-object containment.
3. **Separate Qiskit execution from conversion.** Move Aer execution from
   `interop/qiskit/execution.py` to the Execution Provider-owned location; keep
   `interop/qiskit` for object/format translation. Remove the Runtime reverse
   dependency after a compatibility window.
4. **Move vendor dynamic dialects.** Relocate Braket IQM lowering out of Runtime
   into Execution Provider adapters while Runtime consumes only the approved
   dynamic artifact interface.
5. **Converge text formats.** Make compiler OpenQASM 2/3 and QCIS emitters the
   sole implementations. Convert legacy utility exports into narrow delegates,
   then deprecate/remove them according to Stable Core policy.
6. **Harden extension crossings.** Platform updates the transitional SDK types;
   Ecosystem expands extension conformance without creating another registry.
7. **Add future adapters one at a time.** Add CUDA-Q/QX conversion only after
   their artifact semantics are specified and conformance fixtures prove that
   replacing the adapter does not change consumers.
8. **Keep the retired frontend boundary closed.** The v0.1 `encoding` package
   has been removed; new encoding conveniences must build ordinary
   `fq.Circuit`/`fq.Module` programs rather than recreate a device-side API.

## Format semantics requiring Core unification

Before new adapters or migration work, Core/integration must decide and version:

1. canonical qubit/wire order, tensor-axis order, statevector amplitude order,
   and multi-qubit matrix basis order;
2. classical-bit identity, register flattening, measurement mapping, and
   condition/control-flow representation;
3. parameter identity, expression algebra, binding time, scalar domain,
   trainability, and differentiation ownership;
4. global phase representation and whether semantic fingerprints include it;
5. dtype/precision rules, complex scalar handling, and permitted tensor
   ownership in IR;
6. the separation of circuit instructions from observables, measurement
   requests, shots, seeds, and execution policy;
7. canonical metadata/provenance value types, size limits, unknown-key behavior,
   and serialization validation;
8. custom unitary validation, endianness conversion, labels, calibration data,
   and approximation/fallback reporting;
9. static versus dynamic program capability and the earliest fail-closed stage;
10. provider-neutral execution result, counts/sample ordering, job identity,
    error category, and external result decoding semantics;
11. artifact profile identity/version/media type for OpenQASM, QCIS, future QIR,
    CUDA-Q, and QX forms;
12. compatibility ownership and removal conditions for legacy engine QIR and
    duplicate text exporters.

Until these are approved, adapters must preserve the existing contracts,
report unsupported semantics explicitly, and avoid adding private substitute
IRs or free-form cross-layer dictionaries.

## Verification map

Tests added under `tests/team/ecosystem/` provide:

- dependency-free AST guards for external SDK imports, owned type annotations,
  JAX kernel locality, and the exact known Runtime reverse-dependency debt;
- Qiskit import/export/import semantic fingerprint, global phase and classical
  bit preservation, machine-readable lossy rejection, and recursive imported
  object containment;
- PennyLane complex128 state equivalence, import/export/import fingerprint,
  combined failure-feature reporting, and recursive imported object
  containment.

The repository's existing Qiskit/PennyLane contract and conformance suites
remain authoritative for full opcode matrices and supported dependency lanes.
OpenQASM/QCIS exact text, parse semantics, failure diagnostics, and hash
determinism remain covered by `tests/internal_ir/test_phase2_text_emitters.py`
and `tests/internal_ir/test_phase2_offline_deployment.py`; duplicating those
Compiler-owned fixtures here would create a second test specification.

## Execution evidence

Executed on 2026-09-03:

- team scope preflight and branch-diff scope check: passed;
- `tools/check_architecture.py`: passed;
- focused boundary/contract suite: 36 passed;
- Qiskit 2.5.2, Qiskit Aer 0.17.2, and PennyLane 0.45.1 real interop suite:
  51 passed, including the new team tests;
- Ruff and Black over `tests/team/ecosystem/`: passed;
- `tools/ci_tier.py pr-default`: 1807 passed, 12 skipped, 3 failed in the
  first container run. Two failures were caused by the linked-worktree Git path
  not being mounted and passed after an absolute-path remount. The remaining
  pre-existing `tests/internal_ir/test_performance_budget.py::test_approved_import_verify_budget_is_machine_enforced`
  timing gate failed twice in the container and was not retried again under the
  repository no-repeat policy. It is unrelated to the documentation and
  Ecosystem tests in this change, but the default gate is therefore not
  reported as wholly green.
