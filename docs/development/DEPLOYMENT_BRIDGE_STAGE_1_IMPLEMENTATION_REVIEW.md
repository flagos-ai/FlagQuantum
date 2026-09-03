# Deployment Bridge Stage 1 implementation review

Updated: 2026-09-03

## Result

Stage 1 implements a private, explicit and offline compatibility inspector for existing
`DeploymentPackage` values. It determines only whether a package is eligible to enter a
future verified recompilation flow and returns an immutable, deterministic report.

The result deliberately says `eligible_for_verified_recompile`, not “compatible.” A
positive report still requires separately approved import, legalization, artifact
sealing and execution work. This stage creates no `TargetIR`, executable artifact, job
or provider request.

The checker recognizes OpenQASM 2, static OpenQASM 3 and QCIS profiles; checks the legacy
package validator, target profile, gate set, capacity, shots, topology and calibration
snapshot completeness; and fails closed on malformed, sensitive or oversized metadata.
Reports contain only closed findings, hashes, sizes and canonical profile information.
Raw programs, IR, metadata, provider/backend labels, credentials and exception text are
not returned.

There is no registration, environment activation, provider SDK, credential access,
network call, background work, telemetry, conversion, execution, submission, public
export or default-path integration.

## Evidence

- Thirteen authorization and behavior tests cover all three positive profiles, the closed
  taxonomy, immutability, mutation isolation, malformed inputs, privacy, limits,
  semantic identity and cross-hash-seed determinism.
- Anonymous provider/backend fixtures contain no real backend, job or credential data.
- Public exports and the existing deployment/provider modules remain unchanged.
- The independent 10/100/1K/10K CPU baseline is recorded in
  `contracts/deployment-bridge-stage1-performance-baseline.json`.
- At 10K explicit inspections, observed p95 was 672.844309 ms and peak traced host
  memory was 254,398 bytes.

The measurements are environment-specific observations, not a public SLA. Proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/deployment_bridge_stage1_performance_budget_candidate.json`.

## Remaining closed surfaces

Stage 1 does not authorize package conversion, TargetIR or artifact creation, execution,
submission, provider SDKs, network access, credentials, real provider/backend/job
identities, shadow or dual execution, canary activation, telemetry, public/default API
changes, deployment migration, legacy retirement, Stage 1 exit or Phase 4 entry.

## Review decision requested

Approve only the private Stage 1 performance budget and its future machine regression
gate with:

```text
approve DEPLOYMENT-BRIDGE-STAGE1-PERFORMANCE-BUDGET
```

Stage 1 exit and every later deployment capability remain separately gated.
