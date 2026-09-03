# Deployment Bridge Stage 1 entry proposal

Updated: 2026-09-03

Status: **ready for owner review — implementation not authorized**

## Objective

Stage 1 proposes a private offline checker that answers one narrow question:

> Can this existing `DeploymentPackage` safely enter a future verified Phase 3
> recompilation flow, and if not, why?

It does not convert a package into `TargetIR` or `SealedExecutableArtifact`. It does not
execute or submit anything, import a provider SDK, access credentials, call a network,
activate shadowing, or modify the package.

The machine-readable scope is
`contracts/deployment-bridge-stage1-entry-proposal.json`.

## Proposed private contract

```text
DeploymentPackage + TargetCapabilities + explicit limits
  -> inspect_deployment_compatibility(...)
  -> immutable CompatibilityReport
```

The report has a closed status:

- `eligible_for_verified_recompile`;
- `requires_target_enrichment`;
- `unsupported`;
- `invalid`;
- `limit_exceeded`.

“Eligible” is intentionally weaker than “compatible.” It means only that a later,
separately approved flow may attempt verified import, legalization and artifact sealing.
It never means artifact-ready, executable, submitted or production-compatible.

## Inspection rules

The checker must:

1. invoke the existing fail-closed `DeploymentPackage` validator without modifying it;
2. recognize QASM/QCIS payload profiles and hash bytes without returning raw program text;
3. require verified import and recompilation rather than casting `CircuitIR` to TargetIR;
4. compare program profile, capacity, shots, topology evidence and target snapshot
   completeness;
5. treat provider/backend names as execution locators, excluding them from report and
   compiler identities;
6. emit only closed findings and a deterministic canonical report identity.

The report may contain digests, sizes, a candidate artifact profile, target fingerprint
and closed findings. It must not contain raw program text, raw IR, raw metadata,
provider/backend labels, credentials or exception text.

## Resource and privacy boundary

Every invocation requires explicit immutable limits for program bytes, metadata entries,
metadata depth, encoded metadata bytes and finding count. Exceeding a limit fails closed.

No environment activation, import hook, registry, background worker, telemetry, network
or provider SDK is allowed. Anonymous fixtures must cover sensitive metadata and ensure
no rejected value leaks into reports or exceptions.

## Required evidence

- Positive OpenQASM 2, OpenQASM 3 static and QCIS fixtures.
- Invalid schema, identity, routing, payload, profile, capacity, shots, dynamic-program
  and privacy cases.
- Mutation isolation and deterministic identities across Python hash seeds.
- Nonsemantic provider/backend labels do not change report identity.
- Semantic program, target and eligibility changes do change report identity.
- Public API snapshot, `DeploymentPackage` schema and default execution remain unchanged.
- Independent 10/100/1K/10K CPU baseline before any regression budget is proposed.

## Authorized files if approved

Implementation is restricted to one private checker module, its anonymous fixtures and
internal tests, one baseline benchmark, an implementation review and machine-readable
contracts. Existing deployment/provider modules cannot be modified.

## Rollback

The checker has no registration or side effect. Rollback deletes the private module and
its evidence; no user package, stored receipt or deployment path changes.

## Requested decision

Approve Stage 1 implementation and its measurement-only baseline with:

```text
approve DEPLOYMENT-BRIDGE-STAGE1-ENTRY
```

The command does not approve a performance budget, conversion, execution, provider work,
canary, migration, public/default integration, legacy retirement or Phase 4 entry.
