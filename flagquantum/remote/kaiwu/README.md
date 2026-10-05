# Kaiwu remote boundary

This package owns QBoson control-plane concerns: Kaiwu license initialization,
SPQC project selection, task submission, status and result retrieval, recovery,
timeouts, and credential-free receipts.

It must not own QUBO or Ising conversion, precision-policy mathematics,
QDiffusion integration, local tensor execution, or automatic fallback to a
classical solver. Pure conversion belongs in `flagquantum.ecosystem.kaiwu`.

The Kaiwu SDK uses a `user_id` plus an SDK authorization code (`sdk_code`). The
authorization code is treated as a secret. Applications should pass a
`KaiwuCredentials` object or set both dedicated environment variables:

```bash
export QBOSON_USER_ID='...'
export QBOSON_SDK_CODE='...'
```

The generic `USER_ID` and `SDK_CODE` names used by some vendor examples are not
read automatically because they can collide with unrelated application state.
No credential value may enter logs, exceptions, checkpoints, receipts, or test
fixtures.

`initialize_kaiwu_license` verifies the documented Python 3.10 runtime and an
explicit SDK version before resolving credentials or invoking
`kaiwu.license.init`. Import and license failures are re-raised through
FlagQuantum-owned exceptions; vendor exception text is discarded because it
may contain authorization values. The function returns only non-secret SDK and
Python version evidence.

`KaiwuSDKClient` is pinned to the documented Kaiwu 1.3.1 behavior. It uses the
SDK's `task_name + ising_matrix` checkpoint identity so polling and restoration
query the same task instead of creating a new identity. It deliberately does
not interpret undocumented `get_task_result` fields. After completion it may
record only a value-free schema of that documented result dictionary to support
review of a real response. As a result, current SDK documentation is sufficient
for idempotent execution but not for a provider task ID or provider-reported
target; hardware acceptance remains closed until a real pinned response
establishes those mappings.

Before license initialization, the client requires the configured checkpoint
directory to be an existing private, non-symlink directory. Before the first
SDK task operation, it writes and syncs a complete mode-0600 temporary recovery
bundle, then publishes it without overwriting through an atomic same-filesystem
link. An ambiguous network failure therefore leaves enough non-secret identity
to query the same documented task after restart without exposing a partial
final receipt. Existing bundles are opened without following symlinks and are
reused only when they are private regular files whose task, matrix, mode,
sample count, project, and schema all match. Explicit job `save` and `restore`
use the same private-file rules. The pinned 1.3.1 recovery schema also requires
an aware UTC submission timestamp, exact top-level fields, and absent provider
task/target identities; those identities cannot be injected through a local
checkpoint while the approved SDK mapping remains unavailable. Every SDK
solve, poll, and result path reopens that authoritative recovery bundle and
requires the in-memory receipt to match it exactly. A separately saved job
receipt is therefore resumable only alongside the original checkpoint
directory and cannot override its identity.

The package is not re-exported from `flagquantum.remote` while the Ising task
and result contracts remain under architecture review. In addition to the
credential helpers, its experimental entry points expose a Kaiwu-specific
submit/status/result/wait/save/restore lifecycle. The lifecycle is tested with
an injected fake client and does not import the proprietary SDK. A timeout
never implies cancellation or a replacement submission.

Run the credential boundary checks from the repository root:

```bash
pytest -q tests/team/remote/test_kaiwu_credentials.py
```
