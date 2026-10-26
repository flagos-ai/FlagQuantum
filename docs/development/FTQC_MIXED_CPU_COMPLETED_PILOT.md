# FTS-10A: Completed Bounded CPU Pilot Across Two Campaigns

The specifically approved seven-point continuation completed once after controller
PR #863 review and actual-main admission. Its single non-study supervision probe
and all 28 new workload processes completed without failure, refusal or retry.
The two campaigns now contain 48 complete points: 192 successful workload
processes plus the original one failed warm-up, **193 cumulative attempts**.
The original campaign remains failed; it is not relabeled as successful.

This completes the admitted finite observation matrix, not FTS-10 method
selection, FTS-02/09, general FTQC, capacity or a performance target. The user
has deferred 1.5x/2x targets. No new API, approximation, engine, hardware or
additional recovery campaign is admitted by this report.

## Evidence and campaign identities

- [Original partial report](FTQC_MIXED_CPU_PARTIAL_BASELINE.md) and its index/three
  parts retain 164 completed processes (41 warm-ups, 123 measurements), one
  monitor failure and 27 unattempted slots. The original 663-file inventory is
  unchanged byte for byte. Its collector is also unchanged.
- [Continuation evidence](evidence/ftqc_mixed_cpu_continuation_20261025.json)
  retains the new manifest, all 28 producer/watchdog records, the separate probe,
  raw file digests, seven new three-measurement median groups and the combined
  48-point ledger. Every point names its campaign. No completed original
  observation was rerun or replaced, and no fastest-three selection occurred.
- Numerical producer: original accepted main
  `5cd642a69a0142b7c7e42013851c38dee701a3d5`, retained wheel SHA256
  `9a5fc28e9251f3f3ad780deed4a87e48b92efc920bb41321fd720ecca879a5ed`,
  Apple M5, Python 3.12.14/Torch 2.14.1, one thread, no PYTHONPATH/SDK overlay.
  Installed package bytes and runtime/CPU identity were checked before launch.
- Controller: reviewed actual main
  `8d85a602622f124a0b88db370cf6032d25ae2d7d`, script SHA256
  `a8e62ab27f8e989edaea25cb1ea10473df14b76c2fa95bad1e979e1ed4c18880`.
  Watchdog identity is unchanged from the original campaign. Latest report-base
  code is not substituted for this frozen numerical producer.
- Only hostname keys and local absolute path prefixes are redacted in public
  projection. Raw measurements, plan/native metadata and numerical values are
  retained. Raw SHA256 digests identify unredacted originals.

The original failed n8/t4/ring/TN warm-up exceeded the 50-ms host `ps` query
budget. Its monitor termination and absent producer remain in the original
ledger. The reason for the slow query remains unknown. New successful queries
do not demonstrate an environment repair or guarantee later availability.

## New continuation results and budget

| Item | Retained result |
| --- | --- |
| Supervision probe | One owned idle Python process; completed, 11 positive RSS samples, max gap 0.02950566599611193 s. No study circuit. |
| New workloads | 28 completed: seven retained host warm-ups and 21 independent measured processes. No failures, refusals, retries or remaining new slots. |
| Whole continuation wall | 90.11209962502471 s, including admission, probe, startup and cleanup; below 900 s. Entire campaign was shorter than each 120-s workload cap. |
| New raw directory | 964771 bytes. |
| Original plus new raw directories | 5201507 bytes, below 128 MiB; launch free space 3178496000 bytes, above 512 MiB. |
| Maximum new sampled workload-group RSS | 242728960 bytes, below the unchanged 2-GiB stop threshold. Shared pages may be counted more than once. |
| Maximum new workload sampling gap | 0.03888745800941251 s, below unchanged 100 ms. Query timeout remains 50 ms. |
| Maximum new independent error | 3.1086244689504383e-15; all four error fields at every new point pass approved atol=1e-12/rtol=0. |

All seven new groups use n=8: T=4/ring/TN, then T=8/chain and ring across existing
forced exact CPU statevector/MPS/tensor-network paths. Each has exactly one
retained host warm-up and three fresh measurements. The old/new mapping is
explicit: original slots 164 through 191 are referenced, never edited.

### New-point phase medians

Values below are seconds rounded for display; JSON retains exact values. These
are three-process pilot medians, not tail estimates or confidence intervals.

| T | Pattern | Mode | Construction | Planning | Execution |
| --- | --- | --- | --- | --- | --- |
| 4 | ring | tensor_network | 0.00935346 | 0.08385842 | 2.89427238 |
| 8 | chain | mps | 0.00905162 | 0.08298321 | 0.13257125 |
| 8 | chain | statevector | 0.00921842 | 0.08332625 | 0.02658408 |
| 8 | chain | tensor_network | 0.00932004 | 0.08629096 | 2.96117846 |
| 8 | ring | mps | 0.00933883 | 0.08788404 | 0.13545092 |
| 8 | ring | statevector | 0.01008112 | 0.09477100 | 0.03418733 |
| 8 | ring | tensor_network | 0.00975808 | 0.09076033 | 3.34445479 |

Construction/plan/run phase boundaries are the unchanged observer's boundaries:
initial imports, reference/density conversion/error reduction and report I/O are
excluded; lazy imports inside a call are included. Phase sum is not end-to-end
wall time. RSS samples include validation and are not exclusive workspace or a
capacity result; density/probability storage sizes remain separate fields.
The new controller retains whole-campaign wall time and watchdog elapsed times,
but **does not retain a per-workload outer wall duration**. Do not fabricate that
missing metric or relabel watchdog elapsed time as the old outer-wall field.

The campaigns ran at different times with distinct host/cache histories. Their
point sets are disjoint. Preserve those identities when reading the combined
ledger; no combined speedup, final strategy ranking or large-scale extrapolation
is inferred. A fixed pilot with three repeats per point does not settle method
selection or characterize variability.

## Controller mainline admission

PR #863 actual main8d85 passed
[complete CI, all 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40177334386),
[source/single-GPU/required two-GPU checks](https://github.com/flagos-ai/FlagQuantum/actions/runs/40177413416)
and [main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40177294565).
Three original optional pools were unselected. Package job 121120818279 passed
real QEC optional installation and evidence retention. Actual-main focused tests
passed 133 cases with no skips, including controller/watchdog/reachability,
original frozen retirement contracts and observer checks. That source test
profile used development Python3.12.14/Torch2.13 with the explicitly existing
SDK overlay, not a fresh optional profile.

Fresh isolated sdist/wheel-from-sdist preserves 833 package Python members exactly.
New core Python3.12.14/Torch2.14.1 dependencies, native Bell/master/HTH/CLI and
installed three-route public example/observer CLI passed outside the checkout
without PYTHONPATH or SDK overlays. These engineering checks are not retained
study observations. Actual merge included independent #854/#861 changes, audited
by SHA; the controller's four reviewed paths stayed exact. Later independent
#850 AI work in this report's base is not this study's producer or contribution.

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


## Remaining mission work

FTS-10 method selection still needs a reviewed selection criterion and evidence
for its target task and scale. FTS-02/09 and R01-R15 retain their system gaps.
No automatic rerun or further recovery is authorized. Preserve both raw
campaigns, the original failure, all controller identities and successful point
records. FTS-05 public batch API, approximation, FTQC package boundary, larger
scale/Stage B, domestic hardware and experiments need their separate decisions.

## Post-868 mainline acceptance and next reference decision

Evidence PR #868 merged to actual main
`7720ee678c542c81082b8bd4b719ed6428db03b8` and passed
[complete CI, all 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40183439574),
[source/single-GPU/required two-GPU checks](https://github.com/flagos-ai/FlagQuantum/actions/runs/40183448933)
and [main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40183372095).
Three existing optional pools were unselected. Package job 121139172095 passed
real QEC optional installation and evidence retention. Fresh isolated sdist and
wheel-from-sdist preserve all 833 package Python members exactly. New core
Python3.12.14/Torch2.14.1 dependencies/native Bell/master/HTH/CLI and the installed
public three-mode example/observer CLI passed outside checkout without
PYTHONPATH or SDK overlays. These were engineering checks, not new study runs.

Actual-main public JSON exactly equals the checked projection; both raw
inventories remain unchanged. The metadata field repair preserved 11827 numeric
leaf paths/types/values and the full controller identity, using existing
`controller.source_revision` vocabulary. Original field-failure and independent
unseeded Stim sampling-failure logs remain; the new PR and main CI passed that
existing test without changing its seed, sample count or 0.004 threshold. The
earlier failure's cause remains undetermined. Independent #865/#867 main changes
were audited; this report's five paths stayed exact to reviewed head. No local
numerical suite was relabeled as a new run for the docs/data merge.

The [coherent-branch reference proposal](FTQC_COHERENT_BRANCH_CONFORMANCE_PROPOSAL.md)
specifies the next missing finite correctness question. Its new domain and
comparison values are unapproved; this pilot's completion does not activate it.
