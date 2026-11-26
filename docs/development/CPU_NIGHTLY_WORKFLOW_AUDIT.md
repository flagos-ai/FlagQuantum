# Complete CPU nightly workflow integration

Q-0009 maps the full Qualtran v0.7.0 `.github/workflows/nightly.yaml` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`8c45bc2e3f0b903b1c672e9b720915131d493042`, into
`.github/workflows/cpu-nightly.yml`. The frozen source is unchanged. This is an
Integration-owned workflow change, with no compiler, numerical or public API
change.

## Complete source roles and native adaptations

| Source responsibility | Native owner and behavior |
| --- | --- |
| Weekdays at 11:00 UTC and manual dispatch | Identical cron and explicit dispatch on the independent Nightly CPU workflow; no PR trigger or default required-check change. |
| Read-only workflow permission | Only repository contents read access is requested; checkout credentials are not persisted. |
| Workflow/ref concurrency and superseded-run cancellation | A native workflow/ref group and existing hosted capacity group queue runs. Cancellation is disabled to retain main revision readings under the native release policy. |
| One bounded Linux pytest job | One hosted Ubuntu 24.04 job, bounded at 120 minutes, reuses the existing `nightly` tier. The test step is bounded at 90 minutes. |
| Checkout, Graphviz and Python setup | Native checkout/setup versions follow existing CI. The exact upstream Graphviz action SHA is retained. `dot -V` must succeed before tests. |
| Python 3.10 source interpreter | Native supported CPython 3.12 is selected, consistent with the existing reference lane; lower supported versions retain their ordinary CI owners. |
| Test dependencies and editable source installation | Existing isolated-python and install-cpu-torch actions prepare a job-private CPU environment. Native dev/jax/viz/interop extras and `check_dependency_policy.py --print-test-environment ftqc-reference` retain dependency ownership; `pip check` must pass. |
| Required native references | Preflight checks Qualtran 0.7.0, the actual Kickmix 0.1 Simulator and CPU-only Torch. A missing extension or GPU Torch refuses before the test step. |
| Full serial pytest invocation with duration reporting | Existing `tools/ci_tier.py nightly` retains its entire CPU-safe selection. `-n 0 --durations=10` preserves serial execution/reporting. No slow-case or notebook exclusion is introduced. |

The job uses the existing hosted reference-environment preparation path. It does
not claim the locked offline CPU image includes Graphviz. That image's system
Graphviz availability was unverified, so routing this prerequisite through it
would have invented an environment capability. Existing offline CPU jobs and
their image lock are unchanged. No dependency is installed into the local
workspace for this integration.

The native nightly selection predates this change. It excludes
`benchmark_contract`, `scalability`, `distributed_multinode`, `distributed_accel`
and `gpu`. It includes slow CPU cases, CPU distributed semantics and release
gate tests. This is broad CPU validation, without a hardware, performance or
scalability qualification claim. Test extras and reference versions are existing
native policy inputs, not a duplicate upstream dependency lock. The separate
scientific-lock parents retain their acceptance scope.

## Local execution and retained boundaries

The workflow's actual test command is executed through the unmodified tier CLI
against an isolated nine-case pytest fixture. Four CPU/slow/CPU-distributed/
release-gate cases run, while all five excluded categories are deselected.
The success path returns zero; an injected real pytest assertion failure returns
one through both CLI/subprocess layers. CPU selection and failure propagation
are observed execution, not inferred only from YAML text.

Source fingerprints, full preparation ordering, permissions, runner/CPU limits,
serial arguments and time bounds are verified independently. Existing CI-tier,
resource-budget and package-policy tests cover the reused boundaries. Shell
syntax and embedded preflight Python are checked without installing dependencies
or dispatching a remote job. Strict typing, formatting, lint, Integration scope,
repository language, links, required-check consistency, architecture, internal
IR and protected API checks remain acceptance gates.

This accepts the workflow source integration and its local dispatch semantics.
The complete hosted repository suite has not run in this chat. A future hosted
run's result remains its own evidence; the fixture does not certify every test,
SDK, original scientific environment or runner capacity. It does not accept the
separate PR workflow, external Scorecard publishing, GPU gate or final G3 CI.
No job, release, security upload or merge is triggered by this batch.

### Open frozen-fixture prerequisite

Subsequent full-source PR workflow review found a concrete prerequisite that
the isolated dispatch fixture did not cover. For example,
`tests/unit/test_development_tools.py` reads the sibling `ftqc-reference-env`
tree and the external progress CSV directly. The hosted setup currently installs
packages but does not create that source/metadata layout. Package installation
alone cannot supply those files. Other reference consumers also require Jupyter
template data and qualified tool-profile inputs.

Q-0009 is therefore **implementation in progress**, not accepted as a complete
source/dependency closure. Its published workflow and passing local dispatch
evidence remain valid, but they do not close the missing fixture preparation.
Shared frozen input provisioning must be audited and added before this parent
can be accepted. Do not hide the gap by excluding `ftqc_reference`, silently
skipping original comparisons or treating the isolated fixture as the full suite.
No hosted failure was observed; this gap follows from the actual selected
readers and the preparation steps. The complete PR workflow remains Q-0011.

This procedural module defines no quantum primitive stream, resource model,
complex state, measurement history or private work register. Those remain owned
by the algorithm tests selected by the existing tier.
