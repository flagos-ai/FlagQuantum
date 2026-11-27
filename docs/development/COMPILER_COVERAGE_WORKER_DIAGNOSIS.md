# Coverage worker exits remain unexplained

The selected-routing-plan reuse change remains subject to complete required CI.
An isolated Linux reproduction passes, but it does not explain or replace the
two failed CI executions. No timeout, assertion, coverage threshold, compiler
policy, host setting, or kernel was changed to obtain this result.

## Retained failures

PR #1269 source `c66d630765c2e44b04d93fe4b6952946674934ee` lost a worker
in the complete original Rainbow integration reference in jobs `136949451372`
and `136962899594`, run `44937949070`. Both used runner
`fq-bw1100-16-cpu11` on `bm-srwl-sh-zone9-d-bw1100-144g-1-16`. The worker
exited shortly after its 300-second traceback dump, without an assertion
failure. The second execution was the single diagnosed same-head rerun; no
third identical CI rerun was requested.

Earlier successful main-equivalent job `136927797483` crossed the same
diagnostic interval and passed the node in 344.45 seconds. A traceback timeout
message therefore does not itself establish a test timeout or process kill.

## Isolated exact-source experiment

The [raw record](evidence/compiler-coverage-worker-diagnosis.json) includes
the complete logs, exact-source archive command, diagnostic plugin, bounded
launcher, container state, and initial setup failure. The initial container
could not mount the host's private `/tmp` path through rootless Docker. The
corrected setup used a fresh directory under the slot's shared temporary path;
the test had not started in the failed setup.

The corrected experiment used the same host and slot's rootless engine with
the same locked image:

```text
sha256:b12e3004927655651e3589bc9c263a7277574d66779d3c7713071959a7fd328e
Python 3.12.16; pytest 9.2.0; pytest-cov 7.1.0; xdist 3.8.0
COVERAGE_CORE=sysmon; four workers; one numerical thread
```

It ran this complete node, without shortening the reference environment:

```text
tests/integration/test_synthesis_native_catalog.py::test_original_source_reference_remains_a_test_only_check[tests/fixtures/classiq_qmod/example_references/applications/finance/rainbow_options/rainbow_options_integration_method/comparisons/circuit.py]
```

The isolated container had a four-CPU quota, 8 GiB memory limit, no network,
read-only source and root filesystem, and a 600-second outer budget. Its
launcher emitted phase heartbeats and removed its uniquely named container
after completion. Existing containers and host services were not changed.

All workers reported `faulthandler_timeout=300`,
`faulthandler_exit_on_timeout=False`, and unlimited process CPU/address-space
rlimits. The complete test emitted the 300-second dump and then passed:
321.74 seconds in the test call, 326.36 seconds in pytest. The container exited
zero, with `OOMKilled=False`. Worker-down hooks received no error. Their
immediate process polls returned `None`, so they do not independently certify
individual worker process exit codes. The container result and pytest outcome
are the available successful termination evidence.

## Interpretation and remaining work

This was a different execution context from the CI job: a diagnostic plugin,
direct locked-image interpreter rather than the job's editable-install venv,
explicit container quotas, isolated source mount, and no CI accounting plugin.
It does not reproduce the original failure, isolate the responsible component,
or justify disabling diagnostics or changing coverage cores. Read-only kernel
checks found no matching OOM or segfault record; missing records do not exclude
a signal, especially with fatal-signal logging disabled on the host.

A separate short matrix-loop experiment crossed repeated traceback dumps with
`sysmon`, `ctrace`, and no coverage; all three exited zero. It did not exercise
the complete reference or xdist and cannot establish the original root cause.

The next failing CI execution needs its actual worker exit status or signal
and job resource state retained before attributing the failure to a compiler
change. Current-main synchronization includes the independently admitted CZ
collector change; any new-head CI is a new integration reading, not another
same-head rerun or a claimed fix for these exits. Do not merge until that head's
required checks pass.

The supported user workflow remains unchanged:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

program = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
```

Expected: a legal routed program with restored logical output placement. This
diagnosis changes no public API or capability maturity. It provides no SDK
ranking, performance qualification, hardware, or scalability evidence. Formal
SDKBench qualification remains deferred by the user. No machine was rebooted.

## Separate glued-trees shard cancellation

PR #1291 source `ee767b63f3a1376e91742f424b8444aacd1a4359` has a different
failure: run `44982319380`, job `137086262724`, coverage shard 149, exhausted
the unchanged 60-minute PR limit. Its remaining node was
`tests/integration/test_synthesis_glued_trees_notebook.py::test_wide_full_state_at_original_first_time`.
The node began at 03:57:48 UTC on November 22, 2026; cancellation arrived at
04:55:14. A Python watchdog snapshot at 04:05:12 showed the original
`quantum.evolve(20, 24.0)` statevector execution in fused gate-matrix construction.
One snapshot cannot establish deadlock, computational progress or root cause.

[The complete record](evidence/compiler-glued-coverage-diagnosis.json) retains
the cancelled job and log, three diagnostic container outcomes, source launchers,
native build, resource observations and container cleanup. The original cancelled
container was still running at 05:10; its exact labels identified the failed
run. It entered removal at 05:10:44 while a subsequent CI job was starting.
The diagnostic did not terminate that original container. This exposes a delayed
cleanup interval, not the cause of the earlier test stall.

The first setup failed to write its native editable extension into the mounted
source. The second failed while preserving host ownership during a copy under
rootless container capabilities. Neither ran a test. The corrected setup copied
the same Git source into container tmpfs without preserving ownership and built
its native editable installation offline. It used the same host
`bm-zksg-wx-zone1-d-mc550-64g-2-112`, cpu04 engine, locked image above, Python
3.12.16, four xdist workers, `COVERAGE_CORE=sysmon`, one numerical thread and
explicit four-CPU/16-GiB limits. The launcher bounded build plus test at 720
seconds, retained the first execution and removed its uniquely named container.

The complete, uncropped node passed once: 302.62 seconds in the call and 306.74
seconds in pytest. It compares all `2**20` complex128 amplitudes against the
independent original 200-term, ten-repetition NumPy reference at absolute
tolerance `1e-10`, with no workspace qubits. It crossed pytest's 300-second
faulthandler dump and exited zero with `OOMKilled=False`. Each diagnostic cleanup
reported no remaining container under its exact name.

This is not the full shard or its worker history. The isolated run used pytest's
faulthandler at 300 seconds; the failed CI used `tools.ci.pytest_stack_watchdog`
at 300 seconds and disabled faulthandler. Source location, installation paths and
other runner activity also differ. Shared rootless-engine resource accounting
does not certify per-test memory or host exclusivity. The passing node does not
resolve the cancellation, justify changing timeouts, or authorize merging failed
required checks. No same-head CI retry or host/service restart was requested by
this diagnosis. Keep this incident distinct from the Rainbow worker exits above.
