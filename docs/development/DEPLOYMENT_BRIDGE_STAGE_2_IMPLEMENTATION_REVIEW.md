# Deployment Bridge Stage 2 implementation review

Updated: 2026-09-03

## Result

Stage 2 implements a private, explicit and offline compilation dry-run for validated
legacy `DeploymentPackage` values. It recomputes Stage 1 eligibility and then verifies
the chain through CircuitIR import, provider-free compilation, TargetIR legalization,
canonical profile encoding, ephemeral artifact sealing and independent verification.

The only returned value is an immutable `DeploymentDryRunReport`. TargetIR, canonical
payload and sealed artifact objects remain local to the call and are discarded. No
`ExecutionBinding`, runtime adapter, execution, submission, provider SDK, network,
credential, real backend/job identity, telemetry or default-path integration exists.

The hash-bound Phase 3 artifact-profile implementation remains byte-for-byte unchanged.
Stage 2 consumes the existing Phase 2 emission and accepts it only when the existing
Phase 3 canonical validator certifies exact TargetIR/profile bytes. It introduces no new
QASM/QCIS formatter. Shots remain execution intent and never enter TargetIR or artifact
identity. Package name and provider/backend labels are excluded from portable and
evidence identities.

## Evidence

- OpenQASM 2, static OpenQASM 3 and QCIS anonymous fixtures complete the full ephemeral
  identity chain with deterministic golden evidence and artifact identities.
- Closed status/finding contracts, immutability, mutation isolation, privacy, source and
  compiled operation limits, artifact/evidence limits and deadlines fail closed.
- Provider/package label changes leave portable identities unchanged; program, target,
  calibration and shots affect only their correct identity layers.
- Existing Phase 2 offline compilation and Phase 3 artifact tests remain green.
- Public exports, deployment/provider/runtime sources and default execution are unchanged.
- The independent 10/100/1K/10K-operation CPU baseline is recorded in
  `contracts/deployment-bridge-stage2-performance-baseline.json`.
- At 10K operations, observed p95 was 529.137633 ms and peak traced host memory was
  18,104,757 bytes.

The measurements are environment-specific observations, not a public SLA. Proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/deployment_bridge_stage2_performance_budget_candidate.json`.

## Remaining closed surfaces

Stage 2 does not authorize a performance budget, persistent package conversion, returned
TargetIR/artifacts, execution binding, runtime access, execution/submission, provider or
network access, credentials, real backend/job identities, shadowing, canary activation,
telemetry, deployment migration, public/default changes, legacy retirement, Stage 2 exit
or Phase 4 entry.

## Review decision requested

Approve only the private Stage 2 performance budget and its future machine regression
gate with:

```text
approve DEPLOYMENT-BRIDGE-STAGE2-PERFORMANCE-BUDGET
```
