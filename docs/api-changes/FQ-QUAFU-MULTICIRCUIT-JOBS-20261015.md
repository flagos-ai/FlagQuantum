# Quafu multi-circuit jobs

## Authorization and scope

On 2026-10-15 the API owner replied `do` to the concrete proposal to accept a
circuit list through `fq.submit`, create one Quafu job with ordered child tasks,
and retain one recovery receipt. This authorizes the additive input type, not a
new root export, SDK interoperability dependency, live submission, or release.
The contract change is recorded before the adapter implementation.

`program` additionally accepts `Sequence[Circuit | CircuitIR]` for Quafu. The
initial implementation accepts nonempty, same-width logical circuits with one
full-register counts output and shared shots/target. Other providers, local
compilers and physical mappings reject collection inputs before submission.
Single-circuit behavior, `RemoteJob` methods and `ExecutionResult` are unchanged.

## User contract

`job = fq.submit(circuits, target="quafu:Baihua", shots=1024)` makes one POST,
not one POST per circuit. `job.result().counts[i]` belongs to `circuits[i]` in
FlagQuantum's logical bit order. Shots apply to each circuit: three circuits at
1024 shots request 3072 shots overall. The platform's published task limit is
checked before POST. One-element lists remain collection submissions.

`save`, `restore_job`, `status`, `wait`, and `cancel` address the parent job.
Restoring, querying or fetching never submits. A terminal partially finished
collection is reported as failed, with the original provider status retained;
it is not silently converted into a shorter list of successful results. Missing,
duplicate, out-of-range, malformed or failed result rows cannot produce success.
Pagination preserves sequence indices, not response arrival order.

## Receipt compatibility

The existing `flagquantum.remote-job.v1` fields remain unchanged. Collection
decoding context occupies its optional `native_receipt` mapping, tagged
`flagquantum.quafu-collection.v1`, with ordered per-circuit submission identities,
requested shots, `n_qubits` and a SHA256 over that context. The existing outer
receipt and frozen IR retain their historical field names for compatibility;
the new context and examples use qubit terminology. Restore validates this context
before network access. Existing single-circuit receipts continue to decode as
before. Old readers do not support collection receipts; use the matching updated
client. Receipt hashes detect accidental corruption, not malicious forgery or
proof of hardware execution. Credentials and executable QASM are not serialized.

## Protocol evidence and verification

The platform's [submission guide](https://quafu.com.cn/en/docs/sdk/submit)
documents one task per list entry, in input order, and shots per circuit.
The published `quafu-sdk` 0.2.7 wheel was inspected without installation or
execution solely to confirm HTTP fields: POST `/jobs`, `mode="collection"`,
`circuit_level="logical"`, `circuits=[{"qasm": ...}]`, and `client_ref`;
GET `/jobs/{id}/results?offset=...&limit=...&fields=counts` returns columns
including `seq`, `status`, `counts`, `n_clbits`, `shots_returned`, `bit_order`.
The adapter reuses its HTTP transport and count-order conversion; it neither
imports nor copies SDK implementation code. No retry or legacy fallback follows
a rejected or ambiguous collection POST.

Offline conformance uses asymmetric X-on-q0/q1/q2 circuits, one POST, paged and
reordered rows, partial failure, malformed results, receipt corruption and
separate restoration. It compares returned counts with local `fq.run` counts.
Live platform grouping and hardware availability require separate explicit
acceptance and are not established by offline tests.
