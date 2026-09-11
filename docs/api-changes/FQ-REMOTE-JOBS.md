# Remote job lifecycle

## Authorization and naming

The API owner approved the proposed `fq.submit` job journey in this conversation
on 2026-09-11 (affirmative reply to the concrete proposal). This additive change introduces `submit` and
`restore_job` at the root, and `RemoteJob` in `flagquantum.remote.jobs`.
`submit` starts work, `status` queries once, `result` retrieves without polling,
`wait` explicitly polls with a deadline, and `cancel` requests cancellation.
These names describe operations independently of any provider implementation.
Existing `run` behavior, signatures, and historical API baseline stay unchanged.
Status: authorized implementation candidate, not a released feature.

## Scope and compatibility

Remote submission is a separate user journey from synchronous execution, so two
root operations are justified. A thin job object composes existing Quafu and
Jiuding clients; it owns no simulation or compilation passes. No new dependency,
thread, event loop, service, registry, or automatic retry is introduced.
Quafu initially supports full-register counts, with service-side or explicit
local compilation. Grouped expectations must use `run`; they are rejected before
submission because multiple provider IDs need a separately designed group API.
The owner subsequently required native Jiuding job mode without SSH, using a
project and queue. The authorized candidate adds `project` and `queue` arguments
and rejects `workspace` for new submissions. Jiuding submits an inline circuit
through HTTP, runs the image-installed executor, and retrieves bounded results
through authenticated job logs. It requires an explicit image. Synchronous `run`
retains its resident-workspace behavior. No released API is removed.

A job exposes `id`, `target`, `raw_status`, `status()`, `result()`,
`wait(timeout=300.0, poll_interval=3.0)`, `cancel()`, and `save(path)`.
States are `queued`, `running`, `succeeded`, `failed`, `cancelled`, or `unknown`.
Unknown states and missing job records never imply success. Result retrieval
raises an execution error when unfinished or failed. Wait raises TimeoutError
without cancellation or resubmission. Transport errors propagate; no remote side
effect is automatically retried. Cancel is a request, not proof of cancellation.

`restore_job(path)` reads a versioned JSON receipt, never submits, and reloads
credentials from the current environment. A new receipt stores only identity and
result decoding context. It contains no credentials or executable payload. Files
are created exclusively with owner-only access; callers retain them privately.
This preserves output names, physical mapping, and count ordering across restarts.
Jiuding receipts retain project, queue, experiment and job identities plus log
decoding context. Older development receipts remain readable through their
existing managed-workspace transport; new receipts require no workspace.

## Validation and release

Conformance tests cover both providers, unfinished/failed/unknown states,
nonblocking result access, deadlines, cancellation, save/restore and no duplicate
submission. New contracts record only the authorized additions; the historical
baseline is not regenerated. No deprecation or removal applies. Tests using fake
clients establish lifecycle semantics, not hardware or cluster availability.

A [Quafu hardware check](../development/evidence/remote_jobs_quafu_20260911.json)
completed submission and separate-process restoration for one counts task.
A [native Jiuding A100 job](../development/evidence/remote_jobs_jiuding_20260911.json) also completed with exact two-qubit X counts (1024
shots), using HTTP submission and separate-process log retrieval without SSH.
This is a small correctness check, not a performance or scalability claim.
