# FlagQuantum IR Phase 3 Batch G deployment compatibility proposal

Updated: 2026-09-03

Status: **proposal only — no runtime implementation or canary activation**

## Conclusion

The existing `DeploymentPackage` must remain the authoritative public/provider-facing
boundary until a later, separately approved migration proves compatibility. Phase 3
cannot safely replace it by copying fields or reusing its artifact digest: the two models
have deliberately different ownership and identity semantics.

Batch G therefore defines a staged compatibility bridge, not a bridge implementation.
The machine-readable source of truth is
`tests/fixtures/internal_ir/phase3_deployment_compatibility.json`.

## Why direct conversion is unsafe

`DeploymentPackage` groups program IR, emitted program text, shots, provider/backend
metadata, routing evidence and a deployment artifact digest. The Phase 3 boundary splits
these concerns:

| Existing value | Phase 3 owner | Rule |
| --- | --- | --- |
| QASM or QCIS program | `SealedExecutableArtifact.payload` | Accept only through a recognized canonical profile and recompute all Phase 3 identities. |
| `CircuitIR` | Verified importer → QuantumIR → TargetIR | Recompile; never cast or reuse an identity. |
| `shots` | `ExecutionOptions` / `ExecutionBinding` | Execution-only; must not enter artifact or compilation identity. |
| provider/backend name | Target resolution and execution receipt | Locator-only; must not enter portable compiler artifacts. |
| semantic backend capabilities | `TargetCapabilities` | Normalize to closed canonical values and verify completeness. |
| routing evidence | Compilation/legalization evidence | Revalidate; retain the old digest as provenance only. |
| package name | Operator correlation label | Exclude from Phase 3 semantic identities. |
| provider extensions | Namespaced conformance extension or receipt | Allowlist and route by ownership; reject secrets and ambiguous values. |

The current deployment digest includes name, provider/backend, shots, program and routing
evidence. Phase 3 intentionally separates portable artifact identity from dispatch
options and external execution identity. Treating these digests as interchangeable would
break cache correctness, portability and privacy boundaries.

## Compatibility commitment

Batch G changes no current behavior:

- existing `DeploymentPackage` construction, validation, serialization and provider
  submission remain unchanged and authoritative;
- a Phase 3 artifact cannot be submitted through the public provider path;
- no automatic package conversion, dual execution, shadow execution or backend fallback
  is introduced;
- `fq.run`, `fq.plan`, `compile_for_backend`, public exports and default execution remain
  untouched.

## Staged migration

Only Stage 0 is authorized now.

1. **Proposal and inventory:** freeze mapping, ownership, risks and rollback criteria.
2. **Offline compatibility checker:** after separate approval, inspect anonymous package
   fixtures without submission or mutation.
3. **Explicit local dry run:** after separate performance/privacy approval, produce
   legacy-authoritative differential evidence.
4. **Bounded canary:** after a production readiness decision, use per-request opt-in,
   bounded concurrency and an independent kill switch.
5. **Default-path migration:** only through a public compatibility/release process.
6. **Legacy retirement:** only after a published deprecation window and recovery proof.

No later stage inherits authorization from an earlier one.

## Rollback contract

The rollback authority is the unchanged legacy `DeploymentPackage` provider path. Any
future bridge must be removable at one explicit integration point without rewriting user
packages or stored provider receipts.

Rollback is mandatory for semantic mismatch, identity discontinuity, resource/queue/cost
budget breach, privacy exposure, provider behavior leaking past its adapter, serialized
package incompatibility, or unavailable kill-switch/audit evidence.

## Canary readiness—not activation

FlagQuantum is not canary-ready today. Activation requires, at minimum, per-request
opt-in, legacy authority, correctness/privacy/resource/cost budgets, bounded sampling and
backpressure, provider conformance, reversible target resolution, an evidence-retention
policy, end-to-end identity observability, an incident owner and recovery objective, API
compatibility gates, and a separate production approval.

Current blockers include shallow immutability in legacy mapping fields, incompatible
legacy/Phase 3 identity composition, incomplete capability normalization, absence of an
approved offline checker, and missing production observability and incident controls.

## Batch G exit criterion

Batch G may exit when this proposal and its offline contract tests are independently
accepted. Exit does not authorize implementation or activation. Batch H may then perform
only the aggregate Phase 3 exit review unless separately expanded.
