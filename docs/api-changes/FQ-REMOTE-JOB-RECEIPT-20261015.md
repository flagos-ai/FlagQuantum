# Receipt-aware remote submission

## Authorization and naming

On 2026-10-15 the repository API owner explicitly authorized adding
`receipt: str | Path | None = None` to `fq.submit` in this conversation, with
SDK-owned restoration, configuration validation and duplicate-submission guards.
`receipt` names the existing user-owned recovery file, not a provider-specific
option. This record and its narrow candidate signature precede implementation.
No new root export, automatic retry, live job or release is authorized.

## Journey and additive signature

The existing keyword-only arguments retain their order and defaults; append
`receipt: str | Path | None = None` after `credentials`. Without it, submission
and explicit `job.save(path)` / `fq.restore_job(path)` behave as before.

```python
job = fq.submit(circuits, target="quafu:Baihua", shots=1024,
                outputs=fq.counts(), receipt="three-circuits.receipt.json")
result = job.wait(timeout=300)
```

With a receipt path, an existing complete matching record restores the original
job. It does not submit or poll. A new path reserves a private local journal
exclusively before invoking the existing submit path, then saves the ordinary
receipt and atomically completes the journal. Relative paths use the current
working directory; reuse the same personal persistent path after a restart.

The shared wrapper applies to existing Quafu single-circuit/collection and
native Jiuding circuit submissions. It does not add support for other providers
or Jiuding collections. Future providers must join the existing submit/restore
contract before they can use this journey.

## Request identity and failure behavior

The request digest includes ordered canonical circuit/output identities,
single-versus-collection shape, target, shots, compiler, physical mapping, job
name and applicable compute options (image, devices, workspace, project, queue).
Jiuding's environment-derived project and default device count are resolved
before hashing. Credentials are never hashed or serialized; callers supply
current credentials when restoring. The journal stores only digests, never
circuits, executable code, credentials or raw request options.

Changed configuration, malformed files, incomplete journals, missing counterpart
files or restoration errors fail closed. No error falls back to submission. A
lost POST response, failed receipt save or process interruption leaves a guard
requiring provider reconciliation. This is conservative local exclusion, not
provider-side exactly-once execution. Even failures during preparation may leave
a guard; do not delete it and retry without checking whether submission occurred.

The `flagquantum.remote-request.v1` journal lives at
`str(receipt_path) + ".request.json"`. It contains `schema`, `request_sha256`
and nullable `receipt_sha256`. The last field is null until the ordinary receipt
has been saved; its completed digest binds the exact receipt bytes. Both files
use owner-only permissions. Local hashes detect corruption, not hostile forgery
or proof of hardware execution. A journal without a receipt is never treated as
an unsubmitted request. Concurrent callers cannot reserve the same path twice.

## Compatibility, alternatives and migration

No existing receipt schema, return type, standalone restore behavior or released
baseline changes. An existing manually saved receipt lacks verifiable request
context and is not silently adopted: use `fq.restore_job(path)` for it. A new,
intentional experiment uses a new path after reconciling any previous attempt.
Omitted receipt remains explicitly non-idempotent. No deprecation or removal
applies. Notebook-only helpers were rejected because they expose safety internals
and duplicate the provider-neutral lifecycle already owned by Remote.

The implementation is an authorized additive candidate, still subject to normal
PR/CODEOWNER review; this conversation does not substitute for a GitHub approval.
Executable contracts must cover both providers, configuration mismatches,
corrupt/missing files, concurrent callers, ambiguous submissions, failed saves,
credentials exclusion and the unchanged no-receipt/standalone-restore paths.
Release notes and examples must describe the opt-in behavior without claiming
live acceptance from offline tests.
