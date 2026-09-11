# FlagQuantum IR Phase 3 aggregate completion review

Updated: 2026-09-03

Status: **technical exit candidate — owner approval pending**

## Executive conclusion

Phase 3 has completed the authorized private architecture scope. FlagQuantum now has a
provider-neutral internal boundary from target capabilities and TargetIR through a sealed
executable artifact to execution binding, receipt and result. The same conformance suite
passes a local runtime-plan target, a synthetic QASM target and a synthetic non-QASM
artifact target.

This is an internal technical milestone, not a production migration. None of these types
is public or used by `fq.run`, `fq.plan`, `compile_for_backend`, `DeploymentPackage` or a
provider submission path by default.

## Batch results

| Batch | Delivered | Exit evidence |
| --- | --- | --- |
| A | Immutable target capabilities, closed vocabularies and semantic fingerprint | Canonical identity, mutation isolation, diagnostics, privacy and approved budget |
| B | Private TargetIR and target legalization | Deterministic lowering, legality checks, fail-closed unsupported cases and approved budget |
| C | Artifact profiles and sealed executable artifact | Payload/profile/target/compilation identity chain, tamper rejection and approved budget |
| D | Runtime adapter ABI | Binding/receipt/result lifecycle, terminal semantics, identity continuity and approved budget |
| E | Provider-neutral conformance | Identical ordered checks for local, QASM and non-QASM families; approved budget |
| F | Explicit shadow comparison harness | Legacy authority, closed taxonomy, privacy, bounded breakers, kill switch and approved budget |
| G | Deployment compatibility proposal | Exact current-boundary mapping, staged migration, rollback and canary-readiness review; no activation |
| H | Aggregate review | Authorization, implementation, target-family, performance and public/default evidence bound here |

## Shared target boundary

The three anonymous target families use the same sequence:

```text
TargetCapabilities
  -> TargetIR legality
  -> SealedExecutableArtifact
  -> ExecutionBinding
  -> SubmissionReceipt
  -> RuntimeResult
```

The QASM and non-QASM fixtures are synthetic and offline. They prove architectural
conformance, deterministic identity and failure behavior; they do not prove access to a
real provider, queue, device, credential or production service level.

## Identity and privacy result

Program, target, compilation, artifact, binding, receipt and result identities remain
separate. Provider/backend/job locators are execution-only. Credentials, URLs, queue,
quota and price data do not enter portable compiler artifacts or public evidence.

The shadow harness is private and explicitly constructed. It returns the exact legacy
observation, does not expose the candidate result, stores only bounded hash/size/timing
evidence, and has a thread-safe one-way kill switch.

## Performance result

Approved private CPU performance gates exist and pass for Batches A-F at 10, 100, 1K
and 10K scale points. Batch G adds documentation and offline contract checks only, so it
has no runtime performance budget. These budgets are regression controls for the recorded
environment, not public SLAs and not evidence of QPU or production-cloud performance.

## Deployment and rollback result

`DeploymentPackage` remains the authoritative provider-facing boundary. Its identity
cannot be reused as a Phase 3 artifact identity because it combines name, target locator,
shots, program and routing evidence. A future bridge must recompile or revalidate fields
under their Phase 3 owners and recompute the identity chain.

The approved rollback point is therefore simple: keep the legacy deployment and execution
paths unchanged. No user package or stored receipt needs migration to remove Phase 3.

## Residual limitations

- no public IR, target, artifact, adapter, conformance or shadow API;
- no default-path or production runtime integration;
- no real provider SDK, discovery, credential, backend/job submission or queue handling;
- no canary activation, deployment migration or automatic backend fallback;
- QASM and non-QASM conformance targets are anonymous offline fixtures;
- the compatibility checker and package-to-artifact bridge remain unimplemented;
- legacy deployment mapping fields are not deeply immutable;
- no dynamic ProgramIR, control-flow lowering, QIR, timing, pulse or Phase 4 capability;
- no authorization to retire the legacy compiler or deployment path.

## Exit decision requested

Approve only Phase 3 private technical completion with:

```text
approve IR-PHASE3-EXIT
```

This approval must not authorize public APIs, runtime/default integration, production
shadowing, canary activation, provider/network work, deployment migration, legacy
retirement or Phase 4 entry.
