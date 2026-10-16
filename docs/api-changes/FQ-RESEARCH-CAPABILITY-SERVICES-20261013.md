# API Change: research capability services

## Status

Decision: approved on 2026-10-13 for the experimental namespace contract and
for development of a thin external execution adapter. The user supplied the
exact authorization token as the authority for both the API-owner and
Services-domain-owner roles:

```text
approve API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES
```

This approval records the namespace candidate as the authoritative domain
contract and permits a thin external adapter to call it. It does not by itself
promote capability maturity, publish a release, or authorize MCP tool or
resource registration. Those activation gates remain separate and fail closed.

## Approved decision

The namespace-scoped `flagquantum.services` contract described by
[`API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES.md`](../development/API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES.md)
and its machine-readable companion,
[`research-capability-services-v1-candidate.json`](../../contracts/research-capability-services-v1-candidate.json),
is approved at `experimental` maturity for the following exact scope.

The decision covers three P0 capability identities, six scientific
operations, two schema-catalog operations, closed JSON schemas, structured
errors, scientific validators, tamper-evident Evidence Bundles, and explicit
unavailable-only provider fallback across native, Qiskit Aer, and PennyLane
Lightning implementations.

It does not add a root export or approve remote execution, credentials, QPU
submission, protocol registration, natural-language routing, tenant state, or
durable task control.

## Compatibility

The change is additive inside `flagquantum.services`. It changes no existing
root export, IR version, execution default, remote-provider behavior, or MCP
dependency. Existing VQE calls retain `flagquantum-native` as their only default
provider and retain fallback disabled by default.

The schemas remain candidates until approval. Future incompatible changes to
operation signatures, schema identities, fallback triggers, evidence fields,
or error semantics require another reviewed change.

## Evidence

The approval evidence and owner checklist are in
[`API_CHANGE_PROPOSAL_070_APPROVAL_PACKET.md`](../development/API_CHANGE_PROPOSAL_070_APPROVAL_PACKET.md).
The verified implementation commits named there cover native/fake provider
conformance, real Qiskit/PennyLane scientific parity, route explanations,
MCP boundary parity, and downstream evidence consumption.

## Remaining activation gates

On 2026-10-13 the user also approved the proposed maturity and release plan.
The release-artifact audit on 2026-10-16 found that the published `0.3.0rc3`
and `0.3.0` wheels omit the approved research service API. Consequently neither
is a compatible release, the first compatible version remains unassigned, and
MCP registration remains blocked. See the correction in the approval packet.

These decisions permit preparation of an MCP activation candidate. They
do not prove published compatibility, direct-Python/MCP equivalence, wheel/process behavior, or real
client discovery; those remain mandatory activation evidence. Remote or paid
execution, credentials, and QPU submission remain outside the approved scope.
