# Performance telemetry during existing GPU gates

The one- and two-GPU performance commands can now retain read-only time series
through the existing command watchdog. This addresses a missing observation in
the encoder candidate's GPU latency-variance failures: endpoint device occupancy
does not reveal CPU placement, load, GPU utilization or clocks during warmup
and measurement. The new records are diagnostic context, not a variance waiver
or a finding that host contention caused the historical failures.

## Ownership and behavior

Integration owns this tooling change. `tools/run_with_watchdog.py` continues to
own the child process group, output, deadlines, cancellation and exit status.
The optional `--hardware-samples` path requests JSON Lines observations; the
helper in `tools/hardware_telemetry.py` only collects hardware/process facts.
No compiler, simulation, runtime or SDKBench implementation changes. The
existing benchmark and performance evaluator are unchanged.

Each observation includes wall-clock time, watchdog-relative monotonic time,
collection duration and the owned process-group ID. The sources are:

- GPU index, P-state, SM/memory clocks, GPU/memory utilization, used memory,
  power and temperature from a read-only `nvidia-smi` CSV query. All physical
  GPUs are observed. UUIDs are not requested; `CUDA_VISIBLE_DEVICES` is not
  recorded because it may contain device identifiers. GPU indices identify
  rows only within this host observation, not globally across machines.
- Process ID, group, last processor, lifetime CPU percentage and RSS for members
  of the watchdog-owned process group. The accounting query transiently reads
  numeric host process rows, but unrelated rows are discarded before writing.
  Executable names, arguments and process environments are not requested.
- Current permitted CPU sets for observed members of the owned group, plus
  host 1/5/15-minute load averages. Neither value is an exclusive-host proof.

Only validated numeric accounting fields, supported P-state tokens and explicit
unavailable-value tokens are persisted. Unexpected source rows are discarded
with an explicit diagnostic. Probe exceptions retain their class, nonzero exits
retain their status, and nonempty stderr is indicated without retaining its
contents. Raw stdout from failed queries, stderr and exception messages are not
written because they can include device identifiers or infrastructure details.

The watchdog requests the first sample during command execution and subsequent
samples one second after the preceding collection ends. The GPU query has a
500 ms timeout and the process query a 250 ms timeout. Sampling is synchronous
and may delay
watchdog polling by up to their combined timeout plus local accounting/I/O;
existing hard and no-output budgets are unchanged. Collection timestamps and
durations expose the actual gaps rather than promising an exact 1 Hz cadence.
These observations can miss short transients and do not establish causality.

No clocks, power limits, affinities, drivers or services are modified. The
observer consumes CPU and management-interface work, so timings from an observed
run must retain that condition. The next diagnostic execution must retain the
complete time series alongside the original unchanged benchmark verdict, rather
than rerunning until variance happens to pass.

## Failure and resource behavior

The output is streamed with one record per observation. An existing output path
is refused before command launch. Missing tools, nonzero query exits, denied
accounting, malformed process rows, decoding failures and query timeouts remain
explicit diagnostic fields; unavailable readings never mean idle or clean.
Unsupported CSV values `N/A` and `[Not Supported]` remain explicit. A process
may exit between `ps` and affinity inspection; the exception class is retained.

Failure to write requested observations propagates and terminates the owned
command through the existing cleanup path. Command failures retain their exit
status, and workload timeout remains a failure. Hardware sampling is refused
alongside strict RSS supervision because the latter requires a 100 ms maximum
accounting gap. No RSS limit is relaxed.

The two existing local GPU performance steps write
`hardware-run/one-gpu-telemetry.jsonl` and
`hardware-run/two-gpu-telemetry.jsonl`. The existing always-run artifact upload
retains that directory, including failed runs. Warmup remains 10 blocks, sample
count 30, inner loops 20/2000 for local/distributed modes, and watchdog budgets
900/1200 seconds. The unchanged evaluator still decides performance acceptance.

## Local workflow

This example launches no remote or paid job and requires no GPU. On systems
without NVIDIA tooling, the sample explicitly records its unavailability:

```sh
python tools/run_with_watchdog.py --phase local-diagnostic \
  --stall-seconds 3 --timeout-seconds 5 --diagnostics /tmp/fq-diagnostic \
  --hardware-samples /tmp/fq-diagnostic-samples.jsonl \
  -- python -c "import time; time.sleep(1.2)"
python -m pytest tests/unit/test_hardware_telemetry.py \
  tests/unit/test_watchdog_rss.py \
  tests/unit/test_gpu_workflow_watchdog_policy.py \
  tests/unit/test_hardware_lane_policy.py -q
```

Expected behavior: observations span the owned command, probe failures are
explicit, and command outcomes, deadlines, cleanup and the original benchmark
budgets remain enforced. Real Linux/NVIDIA time-series evidence is still
required before attributing any variance failure or promoting performance.

## Verification scope

Eleven new tests pass, including command failure propagation, timeout cleanup,
observation-write failure cleanup, unavailable/nonzero/timed-out probes, owned
affinity, streaming records, existing-output refusal, strict-RSS incompatibility
and direct script execution without `PYTHONPATH`. The workflow regression
retains the original warmup, repetition, loop and deadline values.

An earlier 58-test combined watchdog/workflow run passed after restoring the
tracked benchmark source omitted by sparse checkout. The final sparse
`pr-default` attempt selected 60 tests: 59 passed, while the pre-existing
`test_children_outliving_launcher_are_terminated` failed because the local
sandbox denied the `ps` accounting command used to confirm an empty group after
termination. A separate direct `ps` invocation confirmed the restriction. No
process-accounting or cleanup assertion was weakened, and that attempt is not a
passing default tier. The initial combined attempt had 56 passes and two missing
benchmark-source fixture failures; those setup failures remain distinct from
the later permission failure.

Black, Ruff, English and local documentation-link checks pass. Strict mypy
passes the two tool files and new test file for their Linux execution target,
using explicit package bases for the repository's namespace packages. Initial
type-check invocations exposed duplicate module naming and macOS-only stub
selection for the Linux affinity call; the checker options now state the actual
module and OS context rather than suppressing errors. Full-source Linux CI and
real GPU diagnostic records remain required before integration or attribution.

## Historical read-only A800 query compatibility

Before the collection-minimization revision, a [host probe](evidence/gpu-telemetry-query-compatibility.json)
executed the exact GPU/process query arguments three times on the Linux A800
host used by the two-GPU lane. All six probes succeeded. GPU query durations
were 229.692, 220.468 and 198.093 ms; process queries were 74.720, 77.870 and
77.094 ms. GPU source answers include all requested columns and eight devices;
process accounting returned 2,593 rows.

These GPU durations leave little margin under the original 250 ms diagnostic
probe limit. The GPU probe now has a 500 ms bound; the process probe keeps its
250 ms bound. This changes no workload deadline or performance threshold.
Synchronous observation can consequently occupy up to 750 ms in external probes,
plus local accounting/I/O, before watchdog polling resumes. Actual collection
durations remain recorded. The three successful probes do not guarantee that
every future query will meet either bound.

The probe only read host state. It allocated no quantum workload and changed no
clock, affinity, power, kernel, driver or service settings. Its caller affinity
and load refer to the SSH inspection process and observation time, not to any
benchmark rank. Device idleness at that endpoint is not evidence about previous
failures or the future measurement window. Actual benchmark-overlapping GPU
telemetry remains a separate requirement. The public archive retains query timings, status, row counts and the source
answer digest. The original answer is retained privately; device identifiers,
process inventories, affinity and host load are omitted from the public evidence.

The exact modified collector also ran in a separate bounded read-only Linux
probe. Three streamed observations completed in 331.541, 329.874 and
325.387 ms, at offsets approximately 0, 1.335 and 2.669 seconds. Each contained
nine GPU CSV lines, 2,562 process rows and three owned-affinity entries, with
no reported probe-error/diagnostic fields. The temporary directory was removed
on exit. The archive retains the exact driver, source digest and a summary with internal
host observations omitted;
raw temporary process inventories are not retained, so this is compatibility
inspection rather than complete raw hardware benchmark evidence. It did not run
the quantum workload or the watchdog itself.

After the diagnostic-bound correction, the telemetry and existing GPU workflow
policy selection passes 38 tests. Existing sparse default-tier and local sandbox
limitations above remain unresolved; this focused result does not supersede
them.


## Collection minimization revision

The original collector retained host-wide process rows and requested GPU UUIDs.
Automatic merge approval rejected publishing that payload through CI artifacts.
The revised collector applies the restrictions above before writing any sample;
negative tests inject UUIDs, unrelated process IDs and private error strings and
verify that none reaches the JSON Lines output. This changes diagnostic content
only; workload commands, deadlines and acceptance thresholds are unchanged.

The A800 timings and source digests above describe the earlier collector. They
remain historical evidence and do not certify the revised query arguments or
parser on that host. Actual workload-overlapping observations remain pending.

The minimized revision passes all 64 selected watchdog/workflow tests, including
15 telemetry tests. The sparse `pr-default` selection also passes those 64 tests
on this attempt; the earlier sandbox accounting failure remains recorded above.
Strict Linux-target mypy passes three Python files, and Black, Ruff, English and
documentation-link checks pass. These local selections do not certify the full
repository or real GPU measurement window. Full-source CI must run on this
revision.


## Failure-path diagnostics

The existing watchdog failure snapshot now uses the same minimized collector.
It no longer requests a host-wide command-line inventory, compute-process GPU
UUIDs or host-wide kernel stacks. `processes.log` and `gpu.log` retain only the
validated owned process/GPU observations and explicit status fields. The snapshot
also retains safe numeric load and affinity observations in
`hardware-snapshot.jsonl`. Prelaunch failures record that no command was started;
secondary snapshot failures retain their error class without replacing the
original watchdog verdict. RSS accounting error details likewise retain classes
instead of raw exception messages.

`rank-stacks.log` explicitly states that kernel stacks and command lines are not
automatically collected. This is a diagnostic limitation, not evidence of clean
collectives or an idle host. Cleanup, timeout and RSS acceptance assertions are
unchanged. A negative test covers both live-group and prelaunch snapshots and
rejects injected identifiers, private stderr and unrelated process IDs from every
written artifact.

The first combined local run of this revision passed 65 tests and failed the
existing orphan-cleanup test because the sandbox denied `ps` and process-group
signaling. The test and thresholds remain unchanged; this is a permission failure,
not a passing combined selection.

With the required process-accounting and child-signaling permissions available,
the unchanged sparse default tier passes all 66 tests, including 17 telemetry
cases. Strict Linux-target mypy passes all three Python files; Black, Ruff,
English, links and whitespace checks pass. This resolves the local permission
limitation for this run without weakening cleanup. Full-source Linux CI and
workload-overlapping GPU evidence remain separate requirements.
