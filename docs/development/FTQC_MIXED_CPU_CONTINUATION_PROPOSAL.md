# FTS-10A: Bounded Seven-Point Continuation Proposal

Status: explicitly approved by the human user on 2026-10-25, controller review
and mainline acceptance required before the one admitted probe and collection.
No continuation process or supervision probe has been launched.
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

## Controller implementation under review

The human answer was "Approve the bounded continuation; review the controller
first." This authorizes the exact budget above, including cumulative 193
attempts, without widening the monitoring policy. It does not authorize another
recovery round after a new failure.

`tools/collect_ftqc_mixed_continuation.py` is a separately named private controller.
The original task-local collector and original campaign remain untouched. This
replacement reads/hashes the watchdog first and retains its complete report;
monitoring failure keeps supervision as the primary cause, with producer state
`absent` or `available`. Only healthy supervision permits producer JSON parsing.
Command/numerical failure, malformed reports and capability refusal stop the new
campaign and preserve all unattempted slots. The controller does not fall back.

Admission checks the original manifest/controller identities, original slot
mapping, retained wheel hash and installed package bytes, exact Python/Torch/CPU
profile and unchanged watchdog hash. It removes PYTHONPATH/SDK overlays from
child environments and fixes numerical thread counts to one. The supplied
controller digest must equal the separately reviewed script; this is an identity
check, not a claim that a command-line argument grants review approval.

The single new output directory is `fts10-cpu-continuation-20261025` below the
existing evidence root. Existing output is refused. Whole wall time includes
admission, the one idle-process supervision probe and cleanup. The outer process
reserves 20 seconds of the 120-second cap for watchdog cancellation/cleanup;
this may stop earlier than the watchdog's unchanged 120-second phase limit.
Storage checks include original plus new evidence with 2 MiB reserved headroom.
Sampled RSS and storage checks can overshoot between observations; neither is a
hard allocator/filesystem quota. Any detected budget breach retains a failure.

After this implementation is reviewed and its mainline accepted, the one admitted
invocation is:

```bash
python tools/collect_ftqc_mixed_continuation.py \
  --evidence-root /path/to/retained-evidence \
  --output /path/to/retained-evidence/fts10-cpu-continuation-20261025 \
  --watchdog /path/to/unchanged/run_with_watchdog.py \
  --reviewed-controller-sha256 REVIEWED_SCRIPT_SHA256
```

This command is not an instruction to run before review. Its expected success
stdout is `completed`, exit 0, with 28 completed new workload entries and one
separate successful supervision probe. Failure prints `failed`, exits 1 and
retains the primary report, new failure/refusal and remaining unattempted ledger.
A missing admission artifact or changed profile starts no probe or study point.
The probe uses an owned idle Python process, not a study circuit. A successful
probe is not a guarantee of future monitoring availability.

Controller tests use fake process/report boundaries only: primary monitor loss
with absent/malformed producer, missing/corrupt watchdog, malformed producer,
numerical error, refusal, cancellation and bounded cleanup, combined storage,
wall/disk preflight, no overwrite, exact slot mapping and one-probe/28-workload
ceiling. They are not supervision-availability or study measurements. No actual
probe or continuation has been executed during controller development.

## Post-860 mainline acceptance

The budget proposal merged to actual main
`8e40de646493321c25b3286a21828ce7d5150b25` and passed
[all 54 complete-CI jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40173499759),
[source/single-GPU/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40173508053)
and [main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40173461083).
Three original optional pools were unselected. Package job 121109219977 passed
real QEC optional installation and evidence retention. Fresh isolated sdist and
wheel-from-sdist preserve 833 package Python members exactly; new Python
3.12.14/Torch 2.14.1 core dependencies, native Bell/master/HTH/CLI, public three-mode
example and observer CLI pass outside the checkout without PYTHONPATH or SDK
overlays. These are engineering checks, not another formal campaign. The three
Markdown merge did not change package/tests/tools, so no local numerical suite
was repeated. The original study producer remains the separately retained main
5cd installation; this documentation acceptance does not replace it.

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
- [x] Obtain explicit approval for this continuation budget and controller scope.
- [ ] Review and qualify the diagnostic-order repair and bounded preflight.
- [ ] Run at most one newly admitted continuation and review its complete ledger.
- [ ] FTS-10 final method selection, FTS-02/09 and R01-R15 system acceptance remain
  open. New API, approximation, larger scale, domestic hardware and experiment
  decisions remain separate.
