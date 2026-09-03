# Deployment Bridge Stage 2 entry proposal

Updated: 2026-09-03

Status: **ready for owner review — implementation not authorized**

## Objective

Stage 2 proposes an explicit, private and offline compile dry-run. Starting from an
existing validated `DeploymentPackage`, it would prove the identity chain through
verified import, private compilation, TargetIR legalization, canonical payload encoding,
artifact sealing and artifact verification.

It would return only an immutable evidence report. The ephemeral TargetIR, payload and
artifact would not escape to callers or runtime adapters. No circuit would execute and
no cloud task would be submitted.

This is **Deployment Bridge Stage 2**, not the “Quafu 最小真实纵向闭环” Stage 2 in the
unified roadmap. The latter remains separately governed.

## Proposed private flow

```text
DeploymentPackage + TargetCapabilities + explicit policy
  -> recompute Stage 1 eligibility
  -> validate unchanged legacy package
  -> verified CircuitIR import
  -> provider-free static compilation
  -> TargetIR legalization with shots excluded
  -> canonical artifact-profile encoding
  -> ephemeral seal + independent verification
  -> immutable DeploymentDryRunReport only
```

The dry-run must recompute compatibility itself; it cannot trust a stale caller-supplied
report. It imports `package.ir` as the source of truth only after the existing package
validator has proven that the legacy QASM/QCIS payload still corresponds to that package.

## Identity and semantic boundary

The legacy deployment digest remains opaque provenance. Source-program, compilation,
TargetIR and artifact identities are recomputed through their approved owners and are
never copied from `deployment_artifact_sha256`.

Package name, provider/backend labels and shots do not enter portable compiler or
artifact identity. Shots are validated as execution intent but Stage 2 creates no
`ExecutionBinding`.

Canonical output need not be byte-identical to legacy text because verified compilation
may normalize, decompose or route the program. Instead, the proof chain requires:

1. legacy payload ↔ package IR through existing package validation;
2. package IR ↔ verified QuantumIR through the approved importer;
3. compiled QuantumIR ↔ TargetIR through closed legality checks;
4. TargetIR ↔ canonical bytes ↔ sealed artifact through exact encoding and verification.

The request performs no runtime statevector or QPU comparison. “Legacy-authoritative”
means the existing package/provider path remains the only real deployment authority.

## Canonical encoder requirement

Stage 2 must not duplicate QASM/QCIS formatting or call provider exporters. The current
artifact verifier already computes the canonical expected bytes internally. Entry would
permit extracting one private canonical encoder in `artifact_profiles.py`, then making
both sealing validation and dry-run consume that same implementation.

## Report, privacy and limits

The report may include only closed statuses/findings, hashes, sizes, the artifact profile,
requested shots, bounded stage timings and a deterministic evidence identity. It cannot
contain raw program text, IR, payload, artifact, metadata, provider/backend/package/job
labels, credentials, diagnostic messages or exception text. Timings are nonsemantic and
excluded from evidence identity.

Every call requires immutable limits for Stage 1 inspection, source/compiled operations,
artifact/evidence bytes, per-stage duration and total duration. Checks occur before and
between stages. There is no shared cache, background work, registry, environment switch,
provider SDK, network or telemetry.

## Failure and rollback

Unsupported, malformed, sensitive, oversized, timed-out or internally failing inputs
return only closed failure evidence and no partial artifact. Rollback deletes the private
orchestrator and canonical encoder entry point. It requires no user-package, stored
artifact or provider-receipt migration because the default path never changes.

## Requested decision

Approve only the private compile dry-run implementation, ephemeral TargetIR/artifact
construction, canonical encoder refactor and an independent measurement baseline with:

```text
approve DEPLOYMENT-BRIDGE-STAGE2-ENTRY
```

This does not approve a performance budget, persistent conversion, returning an artifact,
execution binding, execution/submission, runtime adapters, provider/network access,
credentials, real backend/job identities, shadowing, canary activation, telemetry,
public/default changes, migration, legacy retirement or Phase 4 entry.
