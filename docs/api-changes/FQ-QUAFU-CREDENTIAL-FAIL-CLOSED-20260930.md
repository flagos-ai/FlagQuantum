# Quafu credential presence fails closed before submission

## Decision and authorization

Status: **proposed, not approved.** This document records the problem, the
evidence, and the intended behavior so that the API owner and the remote domain
owner can approve or reject it. No code change is included, and no approval is
claimed. Implementation must not land before that approval.

The change is proposed under `docs/development/PUBLIC_API_PROTECTION.md`,
because it alters two protected behaviors: the capability failure stage
(`PUBLIC_API_PROTECTION.md:83`) and the exception class and actionable error
category (`PUBLIC_API_PROTECTION.md:85`).

## Problem and affected user journey

`fq.run(circuit, target="quafu:<backend>")` is the documented first-contact
journey: `README.md:76-94` shows it in the "Same circuit. Different execution
targets." example and asks the reader to configure the Quafu token first. When no
Quafu credential is configured at all, that prose is the only thing standing
between the user and an unauthenticated submission: the call still sends the
submission request, and the user sees a raw transport error instead of a
statement of the missing prerequisite.

The failure is a **silent protocol downgrade**. `QuafuProvider.submit()`
(`flagquantum/remote/qpu/quafu.py:409`) selects the current task API only when
`self._task_api_is_healthy()` is true, and that helper returns `False`
immediately when `self.api_key` is unset (`quafu.py:204-205`). An unconfigured
credential is therefore indistinguishable from an unhealthy server: the code
falls through to `_submit_legacy()` (`quafu.py:470`) and posts to the legacy SQC
endpoint with `_headers()` (`quafu.py:225-229`), which contributes no
authentication header when `self.credentials.token` is empty. This contradicts
`docs/guides/QUAFU_BACKEND.md:9-11`, which states that `QUAFU_API_TOKEN` "is
required for an authenticated fallback to the legacy SQC platform", and it
contradicts the repository rule that unsupported capabilities fail at the
earliest knowable stage and that silent fallback is forbidden (`AGENTS.md`,
Engineering Decision Principle 9).

The same file already contains the correct guard: `verify()` refuses to proceed
without a token (`quafu.py:214-220`). The defect is that it is the only path that
checks.

### Affected user journeys

1. A first-time user who copies the README example at `README.md:90-93` and has
   not yet read the prerequisite sentence above it.
2. A user whose shell lost `QUAFU_API_KEY` (new terminal, unset `.env`) but who
   still has a hardware target configured in a saved script.
3. A user whose `QUAFU_API_KEY` is present but whose task-API `healthz` probe
   fails for an unrelated reason while the legacy token is also unset. This is a
   genuine fallback situation, so it must remain a fallback; only the
   *unconfigured* case is a defect.

## Evidence

Reproduced on `main` at `d6de3a28a45706b5c2dbab70953faa4d5bf9a75a`, with
`QUAFU_API_TOKEN`, `QUAFU_API_KEY` and `QUAFU_TASK_SERVER_URL` cleared, using the
same transport-injection point the repository's own test uses
(`tests/api_contract/test_quafu_service_compile.py`, which patches the provider
factory in `flagquantum.remote.qpu.execution`). The transport is a spy and the
base URL is a non-resolving one, so **no request left the machine**; the spy
records the request that the provider tried to send and then replies with the
failure an unauthenticated submission is modelled to produce. This is a
characterization of the code path, not a live run against Quafu, and the
diagnosis does not depend on the platform's actual answer — the defect is that
the request is sent at all.

Current behavior per target class:

| Target | Outbound requests | Result observed by the caller |
| --- | --- | --- |
| `quafu:Baihua` (hardware) | 1 `POST /task/run/`, auth header **absent** | `urllib.error.HTTPError: HTTP Error 422: Unprocessable Content` |
| `quafu:sim` | 0 | `RuntimeError: Quafu target 'sim' requires the task API; no equivalent legacy SQC fallback is available` |
| `quafu:all-race` | 0 | same `RuntimeError`, naming `'all-race'` |

Two separate findings follow.

**Finding 1 (hardware, the defect).** With compiler `quarkcircuit` and no
`target_qubits`, `submit()` tests the task-API health, gets `False` because
`api_key` is unset, and silently downgrades to the legacy SQC contract:

```text
POST https://<task server>/task/run/?name=flagquantum_job&chip=Baihua&shots=1024
  Authorization: absent
  token: absent
-> caller receives urllib.error.HTTPError: HTTP Error 422: Unprocessable Content
```

The error carries no statement of the missing prerequisite, is not a FlagQuantum
error category, and gives the caller no way to distinguish "you forgot a
credential" from "the service rejected this circuit". The request is sent to the
configured platform endpoint while the configuration needed to authenticate it is
absent.

**Finding 2 (simulator and routing targets, a misleading message).** These
targets fail closed with no network request, which is correct, but the message
names the wrong cause. The actual condition is that no `QUAFU_API_KEY` is
configured; the message reports that no legacy fallback exists. A user cannot act
on it.

Both the single-target probe and the per-target loop behind the table above are
saved in the maintainer's review workspace as `probe_quafu_failclosed.py` and
`probe_quafu_targets.py`. Those two files are not part of this repository; the
in-repository equivalent is the transport-injection pattern in
`tests/api_contract/test_quafu_service_compile.py` named above, plus the
regression test added under `tests/team/remote/`.

## Alternatives considered

1. **Documentation only.** State in the guide that a credential is mandatory.
   Rejected: it does not satisfy the earliest-knowable-stage rule, and the code
   would still attempt a submission with an absent credential.
2. **Guard the credential at the `fq.run` entry point.** Rejected: `fq.run` is
   provider-neutral and must not learn provider-specific credential names.
3. **Guard inside `QuafuProvider.submit()` before any request, and correct the
   routing message.** Preferred. It is the earliest point that knows which Quafu
   protocol is about to be used, keeps the rule in the owning domain (remote),
   and needs no signature change.
4. **Also wrap transport failures in `UrllibTransport`.** Complementary to 3 and
   proposed separately, because it changes the exception class for every
   provider that shares the transport, not only Quafu. Keeping it out of this
   proposal keeps one concern per change; see "Documentation and tooling impact".

## Old and proposed behavior

Signatures, exports, serialized schemas, result fields and enum/Literal values
are unchanged. Only the failure stage and the exception class change.

Old:

* no credential, hardware target -> assemble package -> outbound `POST` to the
  legacy SQC endpoint with no auth header -> `urllib.error.HTTPError` (HTTP 422)
  escapes to the caller;
* no credential, simulator or routing target -> `RuntimeError` that blames the
  absent legacy fallback.

Proposed:

* no credential, hardware target -> fail before any request with a
  `flagquantum.errors.ValidationError` whose message names the missing
  prerequisite and both accepted sources, matching the wording already used by
  `verify()` (`quafu.py:219`);
* no credential, simulator or routing target -> same `ValidationError`, raised
  for the credential reason rather than the routing reason;
* the legacy SQC fallback remains available exactly as documented when
  `QUAFU_API_TOKEN` is set, including when the task-API probe fails for a genuine
  availability reason;
* no outbound request occurs in any no-credential case, which is the property the
  regression test must assert.

`ValidationError` is chosen because it already means "a user-supplied program,
option, or value is invalid" (`flagquantum/errors.py:10`) and it subclasses
`ValueError`, so callers that already catch `ValueError` for a rejected argument
keep working. It is a `FlagQuantumError`, so the failure becomes an actionable
FlagQuantum category without adding a new type.

## Source and behavioral compatibility impact

* Public signatures, exports, documented defaults, result fields and serialized
  schemas: unchanged.
* Protected behavior changed: capability failure stage (from "after an outbound
  POST" to "before any request") and exception class (from `urllib.error.HTTPError`
  or a routing `RuntimeError` to `ValidationError`). Both are listed as protected
  at `PUBLIC_API_PROTECTION.md:83` and `:85`, which is why this proposal exists.
* A caller that currently catches `urllib.error.HTTPError` around a
  no-credential submission will need to catch `ValidationError` instead. That
  path was never a supported contract: it was an unauthenticated request that
  could not succeed. `ValidationError` subclasses `ValueError`, so
  `except ValueError` is unaffected.
* No credential is read, logged, or included in any message, receipt, result
  metadata or exception text. The message names environment variables only.
* Removing the network side effect of an unauthenticated submission is a safety
  fix: it stops consuming the platform's quota and stops registering a submission
  attempt that cannot be authorized.

## Migration example

No migration is required for correct code.

```python
import flagquantum as fq

# Before: with no Quafu credential configured this raised
#   urllib.error.HTTPError: HTTP Error 422: Unprocessable Content
# after having already attempted an unauthenticated submission.
#
# After: the same call raises a ValidationError before any request is sent.
# The exact wording is the owner's to settle; the intended content is the
# missing prerequisite and both accepted sources, for example:
#   flagquantum.errors.ValidationError:
#   A Quafu credential is required; pass api_key=... or set QUAFU_API_KEY,
#   or pass token=... or set QUAFU_API_TOKEN for the legacy SQC platform.
result = fq.run(fq.Circuit(2).x(0), target="quafu:Baihua", shots=1024)
```

Callers that want to keep the old error handling shape can catch the wider base:

```python
from flagquantum.errors import FlagQuantumError

try:
    result = fq.run(circuit, target="quafu:Baihua", shots=1024)
except (FlagQuantumError, ValueError) as exc:
    raise SystemExit(f"Quafu submission not attempted: {exc}")
```

## Versioning

* **First deprecation version:** not applicable. The previous behavior was a
  defect, not a supported contract; there is nothing to deprecate and no alias to
  carry.
* **Planned removal version:** not applicable.
* **First version containing the fix:** to be assigned by the release owner.
* The change is backward compatible for every call that supplies a credential,
  which is the only case that could previously succeed.

## Documentation and tooling impact

* `docs/guides/QUAFU_BACKEND.md` gains an explicit statement that a submission
  without any Quafu credential is rejected before any request, next to the
  existing credential table. The existing sentence that the legacy token is
  "required for an authenticated fallback" becomes enforced rather than
  aspirational.
* A regression test is added under `tests/team/remote/` (the remote domain's test
  location) that asserts, for `quafu:Baihua`, `quafu:sim` and `quafu:all-race`
  with all Quafu environment variables cleared: `ValidationError` is raised, its
  message names the credential, and the injected transport records **zero**
  calls. A companion test asserts that setting `QUAFU_API_TOKEN` restores the
  legacy fallback, so the fix cannot be satisfied by disabling the fallback.
* No API contract snapshot, capability matrix entry or benchmark artifact is
  regenerated. `fq.run` keeps its signature.
* Out of scope, and deliberately not proposed here: wrapping
  `urllib.error.HTTPError` and `urllib.error.URLError` inside
  `UrllibTransport` (`flagquantum/remote/qpu/http.py:70`, `:80`, `:100`) so that
  transport failures surface as a FlagQuantum category for every provider. That
  touches Azure, Braket, Quafu and the generic provider at once and deserves its
  own proposal and its own approval.

## Owner and approvals

* Owning domain: `remote`. `docs/api-changes/**` is a shared path, and
  `python tools/check_team_scope.py --team remote --files
  docs/api-changes/FQ-QUAFU-CREDENTIAL-FAIL-CLOSED-20260930.md` passes.
* Required approvals before implementation: **API owner** and **remote domain
  owner**.
* Implementation note for the owner: the guard belongs at the top of
  `submit()`, using the same two accepted sources already resolved in
  `__init__` (`QUAFU_API_TOKEN` for `credentials.token`, `QUAFU_API_KEY` for
  `api_key`). The routing message at `quafu.py:443` should be corrected in the
  same change, because it currently reports the routing cause when the
  credential cause is the true one.
