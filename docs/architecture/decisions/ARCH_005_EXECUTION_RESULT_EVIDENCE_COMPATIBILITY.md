# ARCH-005: Execution Result and Evidence Compatibility Boundaries

Status: Proposed

Date: 2026-09-03
Basis: Phase 0 inventories from eight teams; no new or replacement result implementation.

## Context

Stable `ExecutionResult`, Core `ExecutionRecordContract`, Deployment/Target/Dynamic/
Compiler ABI results, and backend results coexist. Runtime evidence, Core
provenance, capability evidence, audit verdicts, and free-form metrics are also
not unified. Numerical results, durable execution records, and release claims
have different responsibilities. Merging them directly can lose bit ordering,
identity, failure, fallback, or the actual execution path.

## Decision Candidates

1. Stable `ExecutionResult` remains the user-facing projection. This ADR neither
   moves its implementation nor promises tensor payload serialization or changes
   existing accessors.
2. Core separately owns a versioned Execution Record/Evidence envelope recording
   request/plan/artifact/attempt/provider identities, actual target/path/device/
   precision, ownership, memory, communication, fallback/degradation, timestamps,
   failure, and namespaced evidence references.
3. Raw measurements, signature material, and audit verdicts are stored separately
   and linked by digests. Audit determines claim eligibility; a Provider or an
   interface cannot establish capability merely by existing.
4. Backend-native results and metrics may remain within their domain. Crossing a
   Provider boundary requires a lossless projection of core fields. Extensions
   use controlled namespaces and cannot overwrite core fields.

The evidence envelope uses ARCH-003's two orthogonal status axes and the levels
`basic < observable < certification`. Every critical fact carries its value,
field exposure status, source, and scope. Support status describes capability;
it does not identify the source of a field. The weakest required evidence limits
the claim level. Missing fields become blockers, not inferred values.

## Prohibited Practices

- Do not create a fourth public result or wrap counts as a unified result while
  omitting wire/bit order.
- Do not treat free-form metadata as Evidence or let native fields overwrite
  failure, fallback, or identity.
- CPU distributed tests, mocks, replicated execution, and interface tests cannot
  support scalability or QPU claims.
- Do not change snapshots to promise new tensor serialization or alter stable
  exceptions.
- Do not interpret `not_exposed`/`unknown` as evidence that something did not occur,
  particularly CPU, host, or backend fallback. Known fallback must appear in
  results and evidence.

## Compatibility

Existing results retain byte and behavioral compatibility. Adapters only add
separate records/evidence. Existing provenance/metrics reading rules remain;
conflicts must be visible. Future stable result fields, type changes, or changes
to summary/diagnostics require a new schema/version, migration fixtures, and an
API Change Proposal.

## Migration Sequence

1. Freeze stable results, Deployment counts, bit order, and legacy evidence fixtures.
2. Define the evidence envelope, failure taxonomy, and contract fake.
3. Connect a Local Simulation adapter, then a Remote/QPU fake adapter.
4. The Runtime attempt coordinator produces consistent success and failure
   records; Audit consumes the same envelope.
5. Retire cross-layer Deployment/Target results only after callers reach zero and
   the compatibility window ends.

## Acceptance Tests

- Stable result behavior and serialization fixtures remain unchanged; core field
  conflicts cannot be overwritten.
- Asymmetric bit order, leading zeros, shot accounting, and failure/cancellation
  terminal states.
- Success, failure, and fallback all produce records with complete identities.
- Local simulation and remote fakes pass the same result/evidence conformance suite.
- Distributed claims fail closed if any required field or real evidence is missing.
- Schema and negative fixtures cover basic/observable/certification transitions
  and observed/declared/not_exposed/unknown/not_applicable combinations. Claim
  level must not exceed evidence level.

## Open Questions

- Execution Record visibility, retention, and privacy redaction rules.
- HMAC/signature key management and artifact URI ownership.
- References for large tensor/sample payloads, metrics namespace registration,
  and the maximum failure-cause chain length.
