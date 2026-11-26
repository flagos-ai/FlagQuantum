# Complete repository Scorecard workflow integration

Q-0010 maps all responsibilities of the frozen Qualtran v0.7.0
`.github/workflows/ossf-scorecard.yaml`, revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`2caefebb3790e012c5ab35d6657b8144248581b5`, into the native
`.github/workflows/security-scorecard.yml`. The original source is unchanged.
This integration defines a workflow artifact; no scan or result publication is
triggered in this chat.

## Complete source roles

| Source responsibility | Native owner and disposition |
| --- | --- |
| Saturday 20:19 UTC schedule and maintainer dispatch | Identical cron/manual events; no PR trigger or required-check change. |
| Default read permissions and credential-free checkout | Explicit contents read access and the existing native checkout version, with credential persistence disabled. |
| Workflow/ref concurrency | Native workflow/ref grouping and existing hosted capacity group; queueing preserves revision readings rather than cancelling an active scan. |
| One ten-minute Ubuntu job | One ten-minute hosted Ubuntu 24.04 job follows existing native runner policy. |
| Scorecard analysis action | Exact original `f49aabe0b5af0936a0987cfb85d86b75731b0186` action commit and all three original inputs are retained. |
| SARIF producer | Exact original `results.sarif` path and `sarif` format. |
| Publish results to OpenSSF | Original `publish_results: true` and job-scoped OIDC permission are retained. This is an external publication when the workflow eventually runs. |
| Upload that result to GitHub code scanning | Exact original upload action commit `b56ba49b26e50535fa1e7f7db0f4f7b4bf65d80d`, produced path and security-events write scope are retained. No separate local archive replaces the produced result. |

Job permissions explicitly include contents read alongside the two write scopes.
GitHub sets unspecified permissions to none when a permissions map is supplied;
this avoids assuming a job map inherits every workflow-level scope. See the
[official workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions).
No write-all token, custom secret, private vulnerability channel or reporting
address is introduced. The existing [security policy](../../SECURITY.md) retains
its ownership and response promises.

## Direct action contracts and retained limits

The immutable
[Scorecard action metadata](https://raw.githubusercontent.com/ossf/scorecard-action/f49aabe0b5af0936a0987cfb85d86b75731b0186/action.yaml)
defines the results path/format and publication inputs. It delegates to a Docker
image tagged `v2.4.1`. Pinning the wrapper commit does not make that transitive
image tag an immutable digest. This source boundary is preserved explicitly;
no scanner-image reproducibility or security qualification is claimed.

The immutable
[SARIF action metadata](https://raw.githubusercontent.com/github/codeql-action/b56ba49b26e50535fa1e7f7db0f4f7b4bf65d80d/upload-sarif/action.yml)
defines the uploaded file and requires security-events write access. The native
job retains the default engine failure behavior: a failed producer prevents the
subsequent upload, without an always-run stale-result path or ignored errors.

Local verification compares the complete frozen workflow and source action
inputs, the exact producer/consumer path, both action pins, schedule, bounded
runner, permissions and unchanged required checks. It does not execute Docker,
OpenSSF publication or the GitHub upload action. A read-only repository probe
found a public repository with default branch main; the analysis endpoint
returned `no analysis found`. That is not evidence of an active scanner, an
existing security review, or a disabled platform feature.

This accepts source-workflow integration and declared data flow. The first actual
scan, service acceptance, uploaded findings, badges, dependency vulnerabilities
and any resulting remediation remain separate evidence. Scorecard practice
checks do not substitute for pip-audit, private vulnerability reporting or a
codebase security audit. No action is dispatched, no feature setting is changed,
and no result is uploaded by this module's local verification. PR #1084 remains
unmerged, with no G3 promotion.

The workflow defines no quantum primitive stream, analytical resource estimate,
complex state, measurement history or work register. Existing algorithm and
runtime owners retain that evidence.
