# FlagQuantum IR Phase 0 Factual Baseline

Status: P0-001/P0-002 inventories complete; IR-001–003 approved by the API owner.
Baseline date: 2026-09-01
Execution plan: [`MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md`](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)

This document records repository facts and Phase 1 compatibility boundaries. It
makes no new IR capability claims and changes no public contracts.

## 1. P0-001: Public Contract Protection Inventory

### 1.1 Current Public Entry Points

These signatures were read with `inspect.signature` in the Docker development
environment:

```text
fq.Circuit(
  n_qubits=None, *, n_wires=None, nqubits=None, bsz=1,
  device=None, dtype=None, inputs=None, config=None
)

fq.CircuitIR(
  n_wires, instructions, version="1.0", dtype="complex64", shape=(),
  observables=(), measurements=(), metadata=<factory>
)

fq.plan(
  program, *, options=None, measurements=None, noise_model=None
) -> ExecutionPlan

fq.run(
  program_or_plan, *, options=None, measurements=None, noise_model=None
) -> ExecutionResult

flagquantum.compiler.compile_for_backend(
  circuit_or_ir, *, coupling_map=None,
  routing_strategy="restore_after_each_gate", optimize=True, config=None
) -> CircuitIR
```

The stable root `__all__` currently has 22 symbols. Phase 0–1 must not add, delete,
reorder, or rename any of them.

### 1.2 CircuitIR Schema 1.0

Authoritative implementation: `flagquantum/core/ir.py`.

Current top-level canonical payload fields:

```text
kind = "flagquantum.circuit_ir"
version
n_wires
dtype
shape
instructions
observables
measurements
metadata
```

Protected behavior:

- `IR_VERSION == "1.0"`.
- Construction fails for versions other than 1.0.
- `n_wires` must be a positive integer.
- Instruction, observable, and measurement wires must exist, be nonnegative,
  unique, and in range.
- Registered opcodes must satisfy arity and parameter requirements.
- Unregistered opcodes require an explicit matrix, channel marker, or dynamic marker.
- Measurement shots, when present, must be positive integers.
- `to_json()` uses ASCII, stable key ordering, and deterministic separators.
- `content_hash` is the SHA-256 of canonical `to_json()`.
- `from_dict()` rejects unknown top-level fields.
- Parameters, parameter expressions, complex numbers, and tensors use current
  versioned encoding rules.
- Objects without deterministic encoding raise `IRSerializationError`.
- `ensure_circuit_ir()` accepts only `CircuitIR` or objects providing `to_ir()`.

Phase 0–1 must not change these behaviors or create public `CircuitIRV1/CircuitIRV2`
classes.

### 1.3 Current plan/run Semantics

The stable execution chain is:

```text
fq.run(program)
  -> fq.plan(program)
  -> validated ExecutionPlan
  -> execute_plan(plan)
```

Protected boundaries:

- `fq.plan` accepts only `Circuit` or `CircuitIR`.
- Caller measurements conflict with existing source IR measurements; they do not
  replace or merge them.
- Stable options may create an explicit sample measurement for a samples target.
- Expectation/amplitudes targets fail closed without corresponding requests.
- Current dynamic instructions do not enter stable `fq.run`.
- Executing an existing `ExecutionPlan` rejects additional options, measurements,
  or noise models.
- `fq.run(plan)` executes the validated plan without silent replanning.
- Requested, selected, and actual backends and fallback visibility continue to
  follow approved contracts.

Internal QuantumIR does not justify changing these rules.

### 1.4 Current Contract Hashes

These hashes detect unintended Phase 0–1 changes; they do not authorize automatic
contract updates:

| File | SHA-256 |
| --- | --- |
| `docs/public_api_v1.json` | `d211967831ced3947445259acb7e5f6557c8fdc4bc560a0978ee2101123e28d4` |
| `contracts/public-api-v0.2-baseline.json` | `ee8f0b7959cc0f758ae14e73f92f1dbbfd4f66d022beadce2cc37f5c4a843dea` |
| `contracts/public-api-v1-candidate.json` | `72124b6557b09ffee1ecbcc682e958d50aa6a1895e77f96d6e3bb63581ddc256` |
| `contracts/experimental-namespace-v1-candidate.json` | `852c74c6441b89764836615ad4aca93eff6b20919e897f1166a1c26635177ba4` |
| `contracts/experimental-surface-v2-candidate.json` | `f400fb5d8de97f4cf51a15e4c8d2634398fbd158b5b0fda0dbfcf0d4afd63ef4` |

Experimental v2 currently has status `implemented_pending_review`. Recording its
hash does not approve or freeze Proposal 011.

### 1.5 Protected Phase 0–1 Files and Surfaces

Without a separate API change proposal and owner approval, do not change:

- Stable exports in `flagquantum/__init__.py`.
- Public types, schemas, or serialization behavior in `flagquantum/core/ir.py`.
- Signatures or stable semantics of `fq.plan`, `fq.run`, or `compile_for_backend`.
- `docs/public_api_v1.json` or approved contracts.
- Protected ExecutionPlan, ExecutionResult, or DeploymentPackage schemas.
- Current measurement conflict rules or plan/run equivalence.

## 2. P0-002: CircuitIR Consumer Matrix

### 2.1 Field Classification

| Field | Current primary use | Proposed internal classification | Phase 1 requirement |
| --- | --- | --- | --- |
| `n_wires` | Construction, validation, memory estimates, backend capacity, emission | Program semantics | Preserve exactly |
| `instructions` | Compilation, planning, simulation, training, drawing, emission, deployment | Program semantics | Preserve order, opcodes, wires, parameters, and matrices exactly |
| `version` | Schema reading and plan serialization | Source schema identity | Retain 1.0; no new public version |
| `dtype` | State representation, precision, planning | Numerical constraint | No implicit precision upgrade/downgrade |
| `shape` | Batch/dense state description | Execution/representation constraint | IR-003 decides its relationship to program identity |
| `observables` | Expectation requests, interop boundaries, execution | Execution request | Extract typed requests under IR-001 |
| `measurements` | Samples/expectations, shots, result ordering | Execution request; explicit terminal operations need separate classification | Preserve conflict and ordering semantics |
| `metadata` | Runtime config, routing, dynamic/channel markers, provenance | Mixed; not wholly commentary | Classified allowlist; unknown semantic entries fail closed |

### 2.2 Component Consumer Matrix

| Component | Reads/transforms | Current key assumptions | Phase 1 compatibility requirement |
| --- | --- | --- | --- |
| `Circuit.to_ir()` | Generates instructions, dtype, shape, batch/runtime metadata; caches output | Does not materialize exponential shapes above 4096 wires | Importer must not force different Circuit generation behavior |
| `Circuit.from_ir()` | Primarily restores instructions/n_wires | Does not fully restore requests/metadata | Do not treat as a complete round-trip oracle |
| Native compiler | Removes/fuses instructions; preserves other fields with `replace()` | Optimization preserves parameter gradients and scientific semantics | No existing Pass migration in Phase 1; differential checks first |
| Routing | Rewrites wires/instructions; writes routing metadata | Physical mappings and evidence currently live in metadata | Classify and validate first; no silent loss |
| Planner | Analyzes instructions, noise, memory, measurements, runtime config | Single measurement source; dynamic paths fail closed | Importer must not rewrite public conflict rules |
| ExecutionPlan contract | Canonicalizes instruction layer order; preserves observables/measurements | Program hash links to executable plan identity | New program identity does not replace current plan identity |
| Stable runtime | Consumes validated plans and executes measurement requests | `run(plan)` accepts no additional options/requests | No default-path changes in Phase 1 |
| Local statevector/density | Consumes opcodes, wires, parameters, dtype, requests | Each backend/dtype has numerical contracts | Compare independently with existing oracles |
| MPS/TN | Consumes interactions, order, observables; may not need dense shape | No silent fallback to statevector | Importer preserves representation semantics |
| Distributed/JAX | Consumes CircuitIR; creates ownership, communication, gradient plans | Distribution semantics must be reported honestly | No Phase 1 scalability claims |
| Noise lowering | Reads instruction/channel metadata; generates IR | `is_channel` affects construction and execution | Include channel metadata in the semantic allowlist |
| Dynamic runtime | Uses `is_dynamic`, `condition`, and other instruction metadata | Separate from stable `fq.run` | Static importer explicitly rejects, never ignores |
| Qiskit/PennyLane interop | Imports/exports CircuitIR; treats observables/measurements as boundary requests | Lossy conversions must be explicit | Retain diagnostics; no broader support claims |
| QASM exporter | Primarily consumes n_wires/instructions | Not a complete OpenQASM 3 importer/compiler | No exporter replacement in Phase 1 |
| QCIS exporter | Consumes static instructions with bound parameters | Unsupported gates/parameters fail closed | Preserve current limits and exceptions |
| Drawer | Projects n_wires/instructions into drawable history | Display only, not a semantic oracle | Keep new IR printers separate from Drawer |
| Deployment | Compiles IR, routing, QASM/QCIS, artifact hashes | DeploymentPackage binds compiled IR and format fields | No schema or submission-chain replacement |
| Module/training | Uses topology, parameters, IR hashes, backend training | Forward and gradients are both contracts | Gradient parity coverage required |
| Correctness/fuzz tools | Generate and minimize CircuitIR | Fixed seeds and opcode schemas | Reuse for corpus tooling, not as the sole golden truth |

### 2.3 Metadata Facts and Blockers

Metadata falls into at least four categories:

| Category | Observed examples | Treatment |
| --- | --- | --- |
| Instruction semantics | `is_channel`, `is_dynamic`, `condition` | Static importer supports or rejects structurally; never ignores |
| Execution/planning constraints | Circuit `runtime_config`, `batch_size`, `logical_state_shape` | Typed import constraints, not arbitrary attrs |
| Compiler evidence | `routing`, `routing_strategy_selection` | Outside source program semantics; retain as provenance/evidence |
| Provenance/debug | Source, audit, nonsemantic labels | Exclude from program semantic hash; define retention policy |

Phase 1 blocker: complete a repository-wide metadata key inventory and assign a
typed destination to every key affecting execution, legality, results, gradients,
or identity. Unknown keys cannot be declared harmless by default.

### 2.4 Confirmed Dependency Direction

```text
public CircuitIR
  -> compilation / planning / runtime / interop / deployment / drawer

future internal importer
  -> may depend on public CircuitIR

public CircuitIR
  -X-> must not depend on future internal IR
```

Phase 1 `_compiler` must not become a dependency of `core.ir` or register root APIs
through import side effects.

## 3. P0-001/002 Acceptance Record

- [x] Read current public signatures in Docker.
- [x] Record CircuitIR schemas, hashes, and fail-closed behavior.
- [x] Record plan/run measurement and dynamic boundaries.
- [x] Record key contract hashes.
- [x] Build the main component/field consumer matrix.
- [x] Identify the risk of treating all metadata as commentary.
- [ ] API owner reviews the factual baseline.
- [ ] Compiler/runtime owners add any missing consumers.
- [x] Complete repository-wide static inventory of consumed metadata keys.
- [x] Complete typed metadata destinations and a second manual audit.

Outstanding reviews do not block continued Phase 0 corpus design, but do block
merging the Phase 1 importer.
