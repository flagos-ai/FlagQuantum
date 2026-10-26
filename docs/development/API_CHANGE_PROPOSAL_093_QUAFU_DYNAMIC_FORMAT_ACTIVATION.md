# API Change Proposal 093: Quafu Dynamic Format Activation

**Status:** Stage A approved and complete; current SDK contract is unsupported

**Owner:** Remote, with Runtime and Deployment contract review

**Public entry point preserved:** `fq.run`

## Decision summary

FlagQuantum must not activate remote DynamicCircuit execution for Quafu merely
because static submission now uses `quafu-sdk`. Activation requires
authoritative, device-scoped evidence that one maintained Quafu program format
preserves the exact mid-circuit measurement, reset, classical feed-forward, and
conditional semantics requested by the user.

This proposal defines the evidence and review sequence. It does not authorize a
network request, credential access, task submission, quota use, charging, or a
claim that any Quafu backend supports dynamic circuits.

The approved Stage A audit is recorded in
[API Change 093: Quafu Dynamic Format Source Audit](API_CHANGE_093_QUAFU_DYNAMIC_FORMAT_AUDIT.md).
Against `QUAFU-Quantum/quafu-sdk` version `0.2.7` at commit
`b54c6e5dadde81e0bba8529a4e03523794fd97c7`, it reached the **unsupported by
the current maintained SDK contract** outcome. Stable execution therefore
remains fail closed. This source-scoped outcome makes no permanent hardware
claim and may be revisited only against newer authoritative evidence.

## Current user behavior

The stable entry already fails closed before provider construction:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError

circuit = DynamicCircuit(2).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

try:
    fq.run(circuit, target="quafu:Baihua", shots=1024)
except CapabilityError as error:
    assert "dynamic_remote_preflight_failed" in str(error)
```

Expected result: no Quafu SDK client is created, no credential is read, and no
task is submitted.

If support is eventually activated, this exact user code remains the entry
point. Users must not construct capability snapshots, provider-legality
evidence, deployment seals, or SDK request objects.

## Why SDK migration is insufficient

API Change 092 resolved transport ownership and physical-program submission:
FlagQuantum can submit a physical program through the SDK with
`compiler="none"`. That proves neither of the following:

- that the service accepts a dynamic program format;
- that the selected QPU implements the requested dynamic semantics; or
- that task results preserve the required classical and measurement data.

OpenQASM 2.0 cannot represent the full FlagQuantum DynamicCircuit contract.
OpenQASM 3, QCIS, or another representation must not be selected by inference.
The accepted format, version, request fields, compiler behavior, and result
shape must come from an authoritative maintained Quafu source.

## Required evidence packet

One activation candidate must bind all evidence to a single provider, device,
environment, service revision, calibration identity, and capture time.

### Program format

- accepted dialect and exact version;
- supported declarations, classical storage, measurement, and reset syntax;
- supported condition syntax, values, widths, and disjunction behavior;
- whether physical qubit indices are accepted with `compiler="none"`;
- whether service compilation may rewrite or discard dynamic semantics; and
- deterministic rejection behavior for unsupported syntax.

### Device capability

- mid-circuit measurement and reset;
- repeated measurement of one qubit;
- classical feed-forward latency and scope;
- condition width, supported values, and permitted conditional operations;
- directed physical connectivity and native operations;
- logical and physical capacity; and
- whether mid-circuit records are returned to the caller.

A coarse `supports_dynamic_circuits` boolean, a basis-gate list, successful
static execution, a calibration screenshot, or an online device status is not
sufficient.

### Request and result contract

- target, shots, compiler mode, physical mapping, and calibration binding;
- idempotency behavior for an ambiguous submission response;
- task and result identity fields;
- final-count bit order and classical width;
- requested and returned shot counts;
- availability and ordering of mid-circuit records; and
- stable error categories for format, capability, precheck, and execution
  refusal.

Raw credentials, authorization headers, and unredacted provider response bodies
must not be retained in the packet.

## Activation sequence

### Stage A: static maintained-source audit

Inspect the reviewed `quafu-sdk` version and maintained service documentation.
Record exact source locations and versions. This stage is local and read-only;
it creates no client and performs no request.

Exit condition: either the maintained sources define a complete candidate
format and request/result contract, or the stage records the exact missing
facts. Missing facts keep the path closed.

### Stage B: maintainer confirmation

Where maintained public sources are incomplete, record confirmation from the
Quafu service owner in a reviewable schema. The record must distinguish
service-wide format facts from device-specific hardware facts.

Exit condition: every required field has an authoritative source, expiry rule,
and device scope. Informal chat, screenshots, and inferred behavior are not
activation evidence.

### Stage C: offline conformance fixtures

Normalize the confirmed facts into the existing
`TargetCapabilitySnapshot`. Reuse API Changes 086, 089, and 090 for preflight,
routing, post-routing rematch, provider legality, package sealing, and result
identity. Do not introduce a second capability model.

Exit condition: positive and negative fixtures prove deterministic admission,
format refusal, semantic preservation, exactly-one mocked SDK submission, and
result normalization without credentials or network access.

### Stage D: stable-facade integration

Wire only the confirmed device and format through the existing `fq.run` path.
Every other Quafu target or unsupported DynamicCircuit feature must retain the
current fail-closed behavior. There is no simulator fallback, static rewrite,
provider substitution, or compiler substitution.

Exit condition: the stable facade reaches the mocked provider exactly once for
the confirmed fixture and remains side-effect free for every refusal fixture.

### Stage E: bounded live validation

Live discovery and live submission are separate approvals. A live-submission
approval must name the target, program, shots, maximum requests, credential
source, quota or cost envelope, retained fields, and stop conditions. An
ambiguous submission is never retried automatically.

Exit condition: one explicitly approved task returns evidence matching the
offline contract. A successful task proves only that exact target, service
revision, program subset, and evidence window.

## Decision outcomes

The audit must end in one of three explicit states:

1. **Unsupported:** maintained Quafu sources state that the required semantics
   or program format are unavailable. Keep stable execution closed and remove
   obsolete positive fixtures.
2. **Insufficient evidence:** the service may support the feature, but one or
   more authoritative facts are absent. Keep stable execution closed and name
   the missing owner and field.
3. **Activation candidate:** all facts are authoritative and the offline gates
   pass. Request separate approval for stable-facade integration and, later,
   bounded live validation.

No result is converted from one state to another by inference.

## Compatibility and ownership

- no stable export, signature, default, exception family, or result schema
  changes;
- `fq.run` remains the only stable execution entry;
- Runtime owns provider-neutral requirement matching and rematch;
- Compiler owns routing and semantic-preservation evidence;
- Deployment owns package sealing and artifact identity;
- Remote owns Quafu source normalization, program legality, SDK calls, and
  result parsing; and
- SDK-native objects stop at the Remote boundary.

## Acceptance criteria

- [x] Stage A records exact maintained-source versions and locations.
- [x] Every program-format field is authoritative or explicitly missing.
- [x] Missing device capability evidence is explicit; no target support is
      inferred.
- [x] Existing snapshot, rematch, seal, and result authorities remain reused.
- [x] Refusal remains side-effect free and deterministically ordered.
- [x] The Stage A PR includes the current public user example
      and expected behavior.
- [x] No live request or support claim occurred.

## Approval requested

```text
approve API-CHANGE-093-QUAFU-DYNAMIC-FORMAT-ACTIVATION
```

Approval authorizes Stage A only: a local, read-only audit of maintained Quafu
SDK and service sources plus a gap record. It does not authorize credentials,
network access, provider requests, task submission, charging, stable-facade
activation, or a live-QPU support claim.

**Approval recorded:** `approve API-CHANGE-093-QUAFU-DYNAMIC-FORMAT-ACTIVATION`
from the API owner on 2026-10-26. The audit inspected the user-identified
`QUAFU-Quantum/quafu-sdk` public repository only.
