# API Change Proposal 009: Framework-Neutral Interoperability Protocol

## Status

**Frozen by API owner — candidate stable contract approved and frozen.**

- Candidate stable namespace: `flagquantum.ecosystem`.
- Root name changes: none.
- Machine contract: `contracts/interop-protocol-v1-candidate.json`.
- Implementation authorization: the API owner requested continuation of
  stabilization review on 2026-09-01.
- Concrete Qiskit/PennyLane adapters remain experimental.
- Approval: `approve 008-010`, explicitly issued by the API owner on 2026-09-01.
- This does not freeze the entire first public alpha API.

## Decision

Stabilize how external quantum frameworks connect, without freezing concrete
implementations for particular third-party versions. The candidate includes:

- `InteropAdapter` and version negotiation.
- An immutable, lazy adapter registry.
- Consistent import/export results and machine-readable conversion reports.
- Strict fail-closed conversion by default; loss requires explicit `allow_lossy=True`.
- Framework-neutral round-trip, rejection, and semantic fingerprint conformance.
- Exception boundaries aligned with `flagquantum.errors.FlagQuantumError`.

`DEFAULT_INTEROP_REGISTRY`, `qiskit`, `pennylane`, and adapter-specific types/functions
are not candidate stable exports. Adding or stabilizing an adapter requires a
separate proposal with a version window.

## Stable Boundary

```text
external framework object
        │
        ▼
experimental adapter implementation
        │  InteropImportResult / InteropExportResult
        ▼
candidate-stable flagquantum.ecosystem protocol
        │
        ▼
versioned CircuitIR
```

External objects cannot enter compiler, runtime, kernel, distributed, or accelerator
layers. Importing `flagquantum.ecosystem`, inspecting registries, or loading adapter
descriptors must not implicitly import Qiskit/PennyLane.

## Changes in This Round

1. `InteropError` joins the stable `FlagQuantumError` hierarchy while preserving
   built-in ImportError, ValueError, and RuntimeError compatibility categories.
2. `run_adapter_conformance` callable defaults become `None`, eliminating process
   addresses from signatures and making contracts reproducible.
3. `flagquantum.ecosystem.__all__` excludes concrete adapters and default registry instances.
4. `fq.experimental.interop` routes only the experimental `qiskit` and `pennylane`
   implementation namespaces.

## Acceptance Criteria

- [x] Candidate stable exports exactly match the machine contract.
- [x] Stable root `fq.__all__` is unchanged.
- [x] Unified error types preserve built-in compatibility.
- [x] Adapters can be imported, discovered, and inspected without third-party dependencies.
- [x] Qiskit 2.0/2.5 and PennyLane 0.44/0.45 adapter evidence continues independently.
- [x] Concrete adapters are not mislabeled stable.
- [x] API owner approved the framework-neutral interop protocol freeze.
