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

- GPU index/UUID, P-state, SM/memory clocks, GPU/memory utilization, used memory,
  power and temperature from a read-only `nvidia-smi` CSV query. All physical
  GPUs are observed; `CUDA_VISIBLE_DEVICES` records the command's selection.
- Process ID, group, last processor, lifetime CPU percentage, RSS and executable
  name from `ps`. Other host processes remain visible to help inspect shared
  load; their arguments and environments are not read.
- Current permitted CPU sets for observed members of the owned group, plus
  host 1/5/15-minute load averages. Neither value is an exclusive-host proof.

The watchdog requests the first sample during command execution and subsequent
samples one second after the preceding collection ends. Each of the two
external probes has a 250 ms timeout. Sampling is synchronous and may delay
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
Unsupported CSV values such as `N/A` remain in the raw source answer. A process
may exit between `ps` and affinity inspection; the refusal is retained.

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
