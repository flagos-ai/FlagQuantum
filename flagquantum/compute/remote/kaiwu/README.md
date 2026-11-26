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
review of a real response. Schema field count, names, shapes, dtypes, and
sequence-type inspection are bounded; unsafe or oversized field names are not
retained, and diagnostic failure does not discard otherwise valid samples. As
a result, current SDK documentation is sufficient
for idempotent execution but not for a provider task ID or provider-reported
target; hardware acceptance remains closed until a real pinned response
establishes those mappings.

## Pinned SDK compatibility record

The public SDK documentation was rechecked on 2026-10-07. This is a contract
comparison, not permission to install, execute, or substitute another SDK
version:

| Surface | Kaiwu 1.3.1 pinned lane | Kaiwu 1.4.1 current documentation |
| --- | --- | --- |
| Python lane | separately approved CPython 3.10 environment | not accepted as a replacement for the pinned lane |
| optimization mode passed to `CIMOptimizer` | string `"quota"` | `TaskMode.OPTIMIZATION`, value `"optimization"` |
| sampling mode passed to `CIMOptimizer` | string `"sample"` | `TaskMode.SAMPLING`, value `"sampling"` |
| sampling count | required for sampling, inclusive range 10 through 2,000 | required for sampling, inclusive range 10 through 2,000 |
| documented recovery identity | exact `task_name + ising_matrix` | exact `task_name + ising_matrix` |
| incomplete result | `solve` documents `None` while the task is running | no equivalent `None` lifecycle contract is documented on the current module page |
| task details | `get_task_result(ising_matrix) -> dict`, keys undocumented | `get_task_result(ising_matrix) -> dict`, keys undocumented |

Sources: the official
[Kaiwu 1.3.1 `cim` module](https://kaiwu-sdk-docs.qboson.com/zh/v1.3.1/source/modules/kaiwu.cim.html)
and the official
[Kaiwu 1.4.1 `cim` module](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/modules/kaiwu.cim.html).
The mode rename and changed solver signature are compatibility breaks at this
boundary. FlagQuantum therefore does not translate modes by guessing the
installed version, and it does not treat the latest documentation as evidence
for the pinned wheel's runtime behavior. Neither version documents the provider
task-ID, provider-target, status, or result-dictionary field names required by
hardware acceptance; those mappings still require an approved, redacted real
response or written provider documentation.

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
Recovery matrices pass through the same strict Remote scalar boundary as new
submissions; JSON booleans and numeric strings are rejected rather than being
coerced into apparently matching floating-point coefficients.
The stored receipt itself is also revalidated through the complete Remote
contract before an SDK operation. JSON floating-point values that compare equal
to integer matrix-size or sample-count fields cannot bypass runtime type checks.
If the first vendor operation fails after the bundle is published, rebuilding
the client with the same task name, matrix, mode, sample count, and project
reuses the original receipt timestamp and recovery path; it must not create a
second local task identity.

Callers may bind an aware UTC `submission_deadline` to the pinned client. A new
`submit` fails through a stable owned error after that instant, before a
recovery bundle or SDK optimizer is created. The deadline does not block
status, result retrieval, or explicit restoration of an already retained task
identity, so resource-snapshot expiry cannot turn recovery into resubmission.

The generic lifecycle independently validates every restored receipt before
calling a client: runtime field types, exact matrix identity, task mode,
mode-specific sample limits, project identity, provider identity strings, and
an aware UTC timestamp are mandatory. Malformed provider status or result
objects also fail through owned `RuntimeError` messages; non-string status,
non-mapping metadata, boolean spins, and non-real energies are never treated as
valid evidence or allowed to surface as incidental attribute/type errors. A
client response that is not a `KaiwuTaskReceipt`, an invalid runtime task-mode
type, and boolean, string, complex, or nonfinite wait controls likewise fail
through FlagQuantum-owned boundary errors.
The pinned client applies the same redaction to result decoding after the
vendor call: exceptions raised while converting or inspecting the returned
array are replaced with a stable Kaiwu error without retaining an exception
chain or vendor-controlled message. Boolean array elements are rejected before
integer normalization, so `True == 1` cannot turn a non-spin provider value
into an apparently valid `+1` sample.
Both save and restore require the receipt's immediate parent to be an existing
private, non-symlink directory; private file bits alone are insufficient when
another user could replace the directory entry. Publication synchronizes both
file contents and parent-directory metadata, and recovery rejects duplicate
JSON object keys instead of accepting an ambiguous last value. Generic receipt
and authoritative SDK recovery JSON is capped at 64 MiB before parsing,
bounding memory use while retaining room for a dense 1,000-spin matrix. The
writer applies the same limit before creating a temporary file, so it cannot
publish a receipt that the recovery boundary must later reject. Reads consume
at most one byte beyond the limit from the already-open descriptor, so growth
after the initial metadata check cannot turn the cap into an unbounded parse.
Private parents, receipt files, and checkpoint directories must also be owned
by the current effective UID; restrictive mode bits on another user's object
are not treated as authoritative recovery state in a privileged process.

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
