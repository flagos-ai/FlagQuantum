# FTS-10A: Bounded Seven-Point Continuation Proposal

Status: proposed resource and recovery admission, not approved or executed.
The [original partial study](FTQC_MIXED_CPU_PARTIAL_BASELINE.md) remains immutable:
164 completed processes, one failed host warm-up and 27 unattempted slots.
Original #846 admission stopped on the first monitoring failure. Neither #859's
merge nor this proposal permits automatic resumption, retry or a monitoring-policy
change.

## Decision requested

After a separately reviewed controller diagnostic repair, authorize **one new
continuation campaign of seven points and at most 28 workload processes**. This
adds exactly one retry of the failed host warm-up and the 27 missing processes.
It changes the cumulative attempted-workload ceiling from the original 192 to
**193**: 165 original attempts plus at most 28 new attempts. Do not describe this
as staying within the old process ceiling. No completed original observation is
rerun, replaced or silently pooled with another campaign.

The whole continuation, including preflight, startup and cleanup, is capped at
15 minutes; each workload remains capped at 120 seconds. This is a request, not
a performance promise. It preserves the existing 2-GiB sampled process-group RSS
stop threshold, 50-ms query timeout and maximum 100-ms sampling gap. Combined
retained evidence from original plus continuation stays at most 128 MiB; require
at least 512 MiB free before startup. RSS sampling can overshoot; it is not an allocator/cgroup limit. No new qubit count, T count, topology,
backend, precision, device or performance target is requested.

## Exact remaining matrix

Every row below has one retained host warm-up and three fresh measured processes,
each constructing once, planning once and executing once. No inner warm-up,
reused-plan timing or automatic retry is admitted.

| n | Authored T | Pattern | Existing exact CPU mode | Original point state | New process maximum |
| --- | --- | --- | --- | --- | --- |
| 8 | 4 | ring | tensor_network | Host warm-up failed; three measurements unattempted. | 4 |
| 8 | 8 | chain | statevector | Unattempted. | 4 |
| 8 | 8 | chain | mps | Unattempted. | 4 |
| 8 | 8 | chain | tensor_network | Unattempted. | 4 |
| 8 | 8 | ring | statevector | Unattempted. | 4 |
| 8 | 8 | ring | mps | Unattempted. | 4 |
| 8 | 8 | ring | tensor_network | Unattempted. | 4 |

Use the original accepted installed observer/circuit implementation from main
`5cd642a69a0142b7c7e42013851c38dee701a3d5`, Python 3.12.14/Torch 2.14.1 core,
Apple M5 CPU, one thread, no PYTHONPATH or optional SDK overlay. Verify wheel,
observer and watchdog hashes against retained admission before launching.
Controller changes do not authorize replacing the numerical producer or current
execution profile. Any missing retained artifact or changed environment stops
admission; do not silently rebuild under different dependencies.

The original deterministic circuit order, output order, forced exact complex128,
no approximation/fallback and independent probability/normalization/density
comparisons atol=1e-12/rtol=0 remain as specifically approved in #846. This is not
an extension of their semantic domain or an adaptive non-Clifford test.

## Root cause known, and what remains unknown

The failed original watchdog retained `rss_monitor_unavailable`: one
`ps -axo pgid=,rss=` query exceeded 0.05 seconds. The owned workload was terminated
with SIGTERM, so there was no producer JSON. The exact reason for the slow host
query is not established. The RSS maximum before loss of monitoring is not a
post-failure capacity measurement. An early successful supervision probe cannot
guarantee that the host scheduler or later queries will meet the deadline.

The local controller read producer JSON before checking the failed watchdog,
therefore reported a secondary `FileNotFoundError`. Correcting that diagnostic
order does **not** repair the monitoring event or justify extending a deadline.
No attempt to disable/relax monitoring, retry `ps` past its timeout, alter the
process-group ownership model or infer a missing producer result is included.

## Required controller repair before collection

Preserve the original collector, its hash, manifest, all files and previous raw
measurements. Introduce a separately named corrected controller and a new output
directory only after explicit admission; review its implementation and tests
before running the continuation.

The corrected private evidence path must:

1. Read and hash available watchdog diagnostics first, including on nonzero
   command return. Preserve reason/detail, command return, sampled RSS/gaps and
   termination evidence as recorded.
2. If supervision failed, classify the primary failure as supervision and retain
   producer report state as `absent` or `available`. Missing producer JSON is
   expected after forced termination, not a replacement root cause.
3. Validate producer JSON and numerical errors only after successful supervision.
   A malformed successful-case JSON is a distinct producer/report failure.
4. Preserve signal/timeout cleanup, first-failure stop and the complete
   refused/failed/unattempted ledger. No retry loop or partial-success exit.
5. Check the whole deadline and combined storage headroom before every launch,
   and retain original-to-continuation point/attempt mapping explicitly.

This changes only a private controller/evidence diagnostic. It proposes no
`fq.*` API, execution-result serialization, numerical engine, benchmark registry,
frame semantics or FTQC package. Do not retrofit these fields into the immutable
old manifest. Tests must cover monitoring failure with absent producer, missing
watchdog, malformed producer after successful supervision, numerical failure,
capability refusal, cancellation, storage/time preflight and no overwrites.

After review, run one bounded supervision-availability probe without a study
circuit, using the unchanged watchdog and a small owned idle process. Record
real process ownership, positive RSS samples, gaps and cleanup. Include that
probe in the 15-minute controller budget; it is not a matrix observation. If
preflight fails, no study process starts. Do not repeatedly probe until it passes.

## Evidence and stopping rules

- Stop on the first correctness, budget, monitor or unclassified command failure.
  Retain the new failure and every remaining unattempted slot. No second recovery
  attempt is included in this request.
- Retain a separate manifest linked to original raw digests and source/wheel/
  controller/watchdog identities. Original failed and unattempted entries remain
  unchanged, even if a corresponding later point completes.
- Preserve the two campaign identities, timestamps and host/cache context. Three
  measurements per completed point give a small pilot median, not a reliable
  tail distribution. Do not mix measurements from different attempts to select
  the fastest three.
- Only if all 28 new processes complete could the combined evidence contain
  48 complete points/192 successful processes plus the original failed attempt.
  That does not retroactively make the first campaign successful or select the
  final mixed-simulation method.
- Phase times, external wall time, native metadata, post-validation RSS and
  probability/density storage retain their existing distinct meanings. No
  capacity, speedup, 1.5x/2x, production or general FTQC claim is admitted.

## Post-859 mainline acceptance

Partial-evidence PR #859 merged to actual main
`d359b776b23b983685ad33d50e34dba708931be9` and passed
[complete CI, all 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40169690485),
[source/single-GPU/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40169692714)
and [main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40169646480).
Three original optional pools were unselected. Package job 121097890320 passed
real QEC optional installation and evidence retention. Fresh isolated sdist/
wheel-from-sdist preserves all 833 package Python members exactly; a new Python
3.12.14/Torch 2.14.1 core installation passes dependencies, native Bell/master,
HTH/CLI and three-route user examples outside the checkout without PYTHONPATH or
SDK overlays. Installed observer smoke is engineering acceptance, not another
formal study measurement. Source/tests/tools are byte-identical to base877, so
no local numerical suite was repeated for the docs/data merge.

The current evidence index and three parts reassemble exactly to the original
complete JSON object: all 192 planned records and 48 summaries, including the
failure and gaps, remain. No file-size gate or fixture was weakened. Original
size-failure logs and lossless repair proof are retained; neither this delivery
nor a green CI run grants the new continuation budget.

## Runnable current user example

Requirements: supported Python, core FlagQuantum and CPU; no optional SDK,
physical device, credentials or dataset. Save as `/tmp/ftqc_interference_example.py`.
Installed command: `python /tmp/ftqc_interference_example.py`; source command:
`PYTHONPATH=. python /tmp/ftqc_interference_example.py`.

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1)
for mode in ("statevector", "mps", "tensor_network"):
    options = fq.ExecutionOptions(
        mode=mode, device="cpu", precision="complex128",
        allow_approximate=False, allow_backend_fallback=False,
    )
    probabilities = fq.run(
        circuit, outputs=fq.probabilities([0, 1]), options=options,
    ).probabilities
    assert probabilities is not None and probabilities.shape == (1, 4)
    print(mode, *[f"{float(p):.8f}" for p in probabilities[0]])
```

Expected stdout:

```text
statevector 0.42677670 0.07322330 0.07322330 0.42677670
mps 0.42677670 0.07322330 0.07322330 0.42677670
tensor_network 0.42677670 0.07322330 0.07322330 0.42677670
```

This existing two-qubit sanity example is not a failed-study-point retry or a
continuation launch. Its success does not establish supervision availability.

## Admission checklist

- [x] Preserve the original monitor failure and all completed/missing evidence.
- [x] Specify a concrete seven-point/28-process ceiling and unchanged monitor
  policy, with the cumulative 193-attempt implication stated explicitly.
- [ ] Obtain explicit approval for this continuation budget and controller scope.
- [ ] Review and qualify the diagnostic-order repair and bounded preflight.
- [ ] Run at most one newly admitted continuation and review its complete ledger.
- [ ] FTS-10 final method selection, FTS-02/09 and R01-R15 system acceptance remain
  open. New API, approximation, larger scale, domestic hardware and experiment
  decisions remain separate.
