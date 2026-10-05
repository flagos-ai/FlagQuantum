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
