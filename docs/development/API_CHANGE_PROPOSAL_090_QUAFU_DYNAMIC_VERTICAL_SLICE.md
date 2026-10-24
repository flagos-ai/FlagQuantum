# API Change Proposal 090: Quafu Dynamic Provider Vertical Slice

**Status:** Proposed; implementation requires explicit API-owner approval

**Owner:** Remote, with Runtime and Deployment contract review

**Scope:** Credential-free contract proposal only

## Decision summary

FlagQuantum should prove one complete dynamic-circuit remote path with Quafu
before generalizing provider abstractions. The slice must reuse the stable
`fq.run` entry point and the existing requirement, snapshot, post-routing
rematch, deployment-package, task, and result authorities. It must not add a
second public capability model or require users to construct evidence objects.

This proposal does **not** claim that Baihua or any other Quafu backend supports
dynamic circuits. A positive target can be activated only after authoritative,
device-scoped facts confirm every required dynamic feature and request-format
constraint. Queue state, calibration cards, basis gates, qubit count, and a
coarse `supports_dynamic_circuits` boolean are not sufficient evidence.

## Current user workflow

The public entry remains unchanged and fails closed before provider activity:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError

circuit = DynamicCircuit(2)
circuit.gate("h", 0)
circuit.measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

try:
    fq.run(circuit, target="quafu:Baihua", shots=1024)
except CapabilityError as error:
    assert "dynamic_remote_preflight_failed" in str(error)
```

Expected result: the call raises `CapabilityError` without importing or
constructing a provider, reading credentials, making a network request,
sealing a package, submitting a task, or consuming random state.

No `fq.QuafuProvider` or `QuafuProvider.from_env()` API is proposed. Provider
construction remains owned by Remote; credentials are consulted only after all
credential-free gates pass and live activation receives separate approval.

## One complete provider path

The vertical slice is:

```text
DynamicCircuit
  -> canonical source requirements
  -> exact Quafu target capability snapshot
  -> pre-routing capability match
  -> offline route and semantic-preservation check
  -> post-routing capability rematch against the same snapshot
  -> Quafu-specific dialect and request legality
  -> immutable deployment-package sealing
  -> exactly-once mocked task creation
  -> normalized result with identity chain
```

No later stage can erase an earlier blocker. A refusal never falls back to a
simulator, static rewrite, another backend, the legacy SQC endpoint, or local
execution.

## Required authoritative facts

Remote may normalize Quafu discovery and calibration payloads into the existing
`TargetCapabilitySnapshot`. Raw payloads and provider SDK objects must remain
inside Remote. A positive snapshot must bind one provider, backend, environment,
target revision, calibration identity, acquisition time, and expiry policy.

The snapshot must explicitly cover:

- mid-circuit measurement and reset;
- repeated measurement and classical feed-forward;
- supported conditional operations, condition width, values, and disjunction;
- whether mid-circuit data is returned;
- native operations and exact directed connectivity;
- logical and physical capacity;
- accepted program dialect and version; and
- provider legality constraints needed to build one request.

Missing facts remain `missing`, `unknown`, `unmeasured`, or `not_exposed`.
FlagQuantum must not infer dynamic support from screenshots, backend status,
qubit count, basis-gate names, calibration recency, or successful static jobs.

## Quafu legality boundary

After API Change 089 rematch succeeds, a private Quafu-owned legality check must
validate the exact routed artifact against the confirmed request contract. It
must cover program dialect, conditional syntax, operation mapping, shots,
target identity, and calibration identity without network access.

The current precompiled Quafu path documents OpenQASM 2.0, while FlagQuantum's
dynamic program contract requires semantics normally represented by OpenQASM
3. A dynamic request therefore remains blocked until the Quafu contract
authoritatively identifies an accepted representation that preserves all
required semantics. Conversion may not delete measurement, reset, classical
storage, or feed-forward behavior.

## Package and submission identity

Package sealing occurs only after preflight, routing, rematch, and provider
legality all succeed. The immutable identity chain must cover:

- source requirement-set identity;
- exact target snapshot and calibration identities;
- routed artifact and routing-evidence hashes;
- provider-legality decision identity; and
- sealed package artifact hash.

The first submission proof is mocked and restricted to the current Quafu task
API. One execution attempt creates at most one POST and carries an idempotency
identity bound to the sealed package. An ambiguous response after POST must not
trigger an automatic retry or legacy fallback. The legacy SQC route remains
blocked for dynamic submission until it has authoritative dynamic-dialect and
idempotency contracts.

## Result contract

The normalized result must preserve provider task identity, exact shots,
counts, bit ordering, package identity, snapshot identity, and calibration
identity. Mid-circuit data is exposed only when the provider contract confirms
it and the response supplies it; FlagQuantum must never fabricate it from final
counts.

## Deterministic blockers

The implementation must preserve API Change 089 blockers and add the following
provider-owned refusal categories:

- `quafu_dynamic_capability_evidence_unavailable`;
- `quafu_dynamic_dialect_unconfirmed`;
- `quafu_dynamic_request_shape_unsupported`;
- `quafu_dynamic_connectivity_unavailable`;
- `quafu_dynamic_provider_legality_rejected`;
- `quafu_dynamic_snapshot_identity_mismatch`;
- `quafu_dynamic_legacy_submission_not_idempotent`; and
- `quafu_dynamic_result_contract_unconfirmed`.

Ordering is deterministic and diagnostics are redacted. No credential, header,
raw payload, or provider response body enters an exception or retained evidence.

## Staged implementation

1. **Snapshot normalization:** normalize fixed, credential-free Quafu fixtures
   into the existing snapshot model and reject incomplete dynamic facts.
2. **Legality preview:** validate one routed artifact and build a redacted,
   side-effect-free request-shape preview.
3. **Offline sealing:** bind the successful gates into the existing deployment
   package without changing public schemas.
4. **Mocked exactly-once submission:** prove one task-API POST, no ambiguous
   retry, no legacy fallback, and normalized task/result identities.
5. **Stable-facade wiring:** keep `fq.run` as the only user entry and retain
   fail-closed behavior for every target lacking complete fixture evidence.
6. **Bounded live discovery:** separately approve one read-only request to
   confirm the exact target facts without retaining secrets or raw payloads.
7. **Bounded live submission:** separately approve one task only after the
   provider contract, cost envelope, target, shots, and result contract are
   confirmed.

Stages 1 through 5 must use fixtures or mocks. This proposal authorizes no
provider SDK import during preflight, credential access, network request, live
discovery, task submission, charging, or hardware-support claim.

## Compatibility and ownership

- no root export, public signature, exception family, or result field changes;
- no Circuit IR, execution-plan, deployment-package, or result schema changes;
- no change to local dynamic execution or existing static remote workflows;
- Runtime owns provider-neutral matching and deterministic admission;
- Compiler owns transformations and routing evidence;
- Deployment owns package identity and sealing;
- Remote owns Quafu payloads, dialect legality, transport, and result parsing;
  and
- Core remains provider-neutral.

## Acceptance

- [ ] The API owner approves this proposal and its staged boundary.
- [ ] Complete fixture facts produce one exact Quafu snapshot.
- [ ] Missing or non-authoritative facts fail closed before provider activity.
- [ ] The same snapshot identity survives preflight and post-routing rematch.
- [ ] Quafu legality validates the exact routed artifact without network access.
- [ ] Package sealing happens only after every credential-free gate succeeds.
- [ ] A mocked execution attempt creates at most one task and never falls back.
- [ ] The result retains the complete identity chain and confirmed bit ordering.
- [ ] `fq.run` remains the only stable user entry.
- [ ] No live support claim or request occurs without separate authorization.

## Approval requested

Approval directive:

```text
approve API-CHANGE-090-QUAFU-DYNAMIC-VERTICAL-SLICE
```

Approval authorizes only stages 1 through 5 as separately reviewable,
credential-free or mocked pull requests. It does not authorize live discovery,
credential use, task submission, charging, or a Quafu dynamic-support claim.
