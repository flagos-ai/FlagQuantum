# FTS-02: Reference Workloads and Measurement Budgets

Status: concrete measurement proposal, awaiting approval of new quantitative commitments. This is FTS-02A, not a completed high-performance simulator or a new public experiment API. The delivery sequence is authorized; the proposed numerical/performance targets are not activated by a documentation merge.

## Baseline-only decision

On 2026-10-24 the user explicitly selected baseline collection first and deferred activation of the proposed 1.5x throughput and 2x capacity targets. [Initial clean-main CPU observations](FTQC_002_CPU_BASELINE.md) reuse existing producers without a new API or performance threshold. This decision does not approve all proposed resource, statistical, approximation or hardware commitments. FTS-02 remains partial.

## Purpose and boundaries

Make the [FTQC master roadmap](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md) measurable before optimizing. Freeze workload definitions, independent references, resource limits, statistical procedures, and timing boundaries. Reuse maintained benchmark infrastructure and delivered user workflows rather than introducing another benchmark registry, numerical engine, decoder API, or evidence hierarchy.

FTS-02 is split into an initial reviewable specification and then a bounded baseline measurement delivery. FTS-02 is complete only when its selected matrix and budgets are approved and clean-revision baseline evidence has been produced and reviewed. Neither this specification nor a small example closes the mission-level scale or experimental requirements.

## Implementation baseline and concurrent work

Main `be4b9d2f347e67a4c3f2eeb095762bb40bb89d2e` passed [complete CI, 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40079493702), [source/single-GPU/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40079501527) and [smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40079443850). Three pre-existing optional manual GPU-pool checks were unselected. The source QEC, retirement-contract and both Simulation fault-signature suites pass with **2,093 passed and one pre-existing inapplicable-case skip**. A fresh isolated sdist and wheel-from-sdist preserve all **818 package Python members** exactly; a clean Python 3.12 / Torch 2.14.1 core installation passes dependency consistency, native/Bell sampling and public examples outside the checkout. This is one fresh core installation, not a claim of a new local 12-profile run. Supported Python/SDK coverage is provided by exact-head CI. Latest proposal base `bfb1fd6f9d042a24515295e311daf15930af9cf2` adds only #800 documentation and has the identical package tree. See [FTS-01 mainline acceptance](FTQC_001_MAINLINE_ACCEPTANCE.md) for exact artifacts and limits.

The separately merged Surface fixed-shot example (#792), AI assessment (#786), private AI consolidation (#795), and pure Dynamic Remote post-routing implementation (#797) are part of this baseline. Reuse them within their documented domains. The annotated Stim experiment bridge contract document #800 has also merged, but its proposed public entries remain unapproved and unimplemented. AI calibration PR #801 is a separate workstream. Review their latest state before creating overlapping contracts or benchmark cases.

The complete-result cross-decoder contract recorded by #794 is still unapproved for implementation. The existing PyMatching Tensor batch prediction and current single-shot decoder APIs are the runnable paths here. The candidate sibling `ftqc` package remains subject to architecture admission; this work creates no package or stable export.

## Workload inventory

The table distinguishes runnable existing workflows, proposed measurements, and external blockers. Unavailable cases remain blocked, never silently omitted or marked successful.

| ID | Frozen input candidate | Existing path / independent reference | Initial measurement scope | Closure / blocker |
| --- | --- | --- | --- | --- |
| W1-A | One-qubit H-T-H, initial zero state, CPU complex128, no noise. | Public `fq.Circuit` and `fq.run` probabilities; analytic `(2 + sqrt(2))/4` and `(2 - sqrt(2))/4`. | Probability error and normalization; construction, planning and execution timings separated. | Runnable current reference. It is not encoded FTQC or a scalable mixed engine. |
| W1-B | Bell and a small GHZ circuit; ideal and selected positioned Pauli faults. | Public circuits and established Simulation/QEC references; analytic parity and independent dense evolution where already available. | Small-state phase/order/readout correctness, not capacity. | Reuse existing independent tests; add only uncovered scenario cases in the later conformance work. |
| W2-A | Repetition d=3, rounds=3, data flip 0.02, measurement flip 0.03, seed=53. | Delivered `examples/qec/stim_pymatching_baseline.py`; actual circuit sampling plus native PyMatching predictions. | Fixed 128-shot smoke, then proposed 4096-shot baseline; batch size 32. | Runnable. The smoke does not support a threshold estimate or precise rare-event comparison. |
| W2-B | Surface d=3/5/7, rounds=3, X/Z readout; data/phase/measurement flips 0.005/0.007/0.009. | The #792 public-user example and its independently checked logical-readout mapping. | Six declared cases, fixed shots and batch size; preserve example-supported distances and readout construction. | Runnable with Stim/PyMatching extras. No invented arbitrary-round option. |
| W3 | Proposed construction distances 3/5/7/9/17, rounds=3; larger distances and longer history separately staged. | Existing `benchmarks/qec_clifford_dem_construction.py` and its independent fault-propagation references. | First measure d=3/5/7. Admit larger points only within approved timeout and memory budgets. | Construction capacity only; it does not establish whole-circuit execution, decoder quality or sharded capacity. |
| W4 | Sparse selections from n=8/16/32/64 and T=0/4/8/16/32 after FTS-10 algorithm selection. | W1-A is the current small interference reference; candidate decomposition/TN methods need separate selection and contracts. | Specify the future matrix now; do not label statevector measurements as delivered mixed-engine acceleration. | Mixed-algorithm selection and implementation pending. |
| W5 | Existing two-qubit measurement/conditional-X/reset workload, fixed seeds and bounded shot settings. | Maintained `dynamic_unified_run` benchmark and native dynamic oracle; analytic feedback/reset invariants. | Reuse existing planning, explicit run, automatic run, native comparison and static sentinel. | Local software control only; no encoded feedback, domestic-device or QPU deadline claim. |
| W6 | Existing compiled circuit sampler; proposed shots=128/1024/4096, batch size=32 initially. | Reuse fixed-shot counts, Wilson intervals and continuous sampling lifecycle from delivered examples. | Trace compilation, sampling, decode and statistics separately; compare total shots at fixed batch before a million-shot run. | Longer runs need the resource-budget gate. Different seed/SDK/ISA/batch combinations are not assumed bit-identical. |
| W7 | Two device-appropriate classes from W2/W4/W6 on one named domestic device. | Existing Compute/Kernels/FlagOS boundaries plus accepted CPU references. | Device identity, operator coverage, precision, transfer and end-to-end time. | Device, software stack, resource owner and allocation are not yet selected. A800 CUDA evidence is not domestic certification. |
| W8 | One licensed experimental dataset with circuits, raw records, mappings, rounds and calibration metadata. | Partner/public raw data, independently reconstructed detectors, and held-out predictions. | Detector means/correlations and logical counts, with separate calibration/validation/test sets. | Dataset, permission and owner pending. Synthetic data or publication figures do not close this item. |

## Independent-reference policy

1. W1-A uses a closed-form interference probability, not a second call to the same simulator. Both normalization and non-Clifford phase effects must be observed.
2. A dynamic same-seed comparison between automatic and explicit execution checks route consistency, not independent physics. Keep the separate analytic measurement/frame/reset checks and established independent references.
3. Reconstructing detector bits from the same DEM matrices checks bookkeeping. Circuit sampling is a separate physical-law cross-check within its supported noise domain; document shared builders and translation assumptions.
4. Matching/BP/BPOSD may differ in tie choices, convergence, or correction fields. Preserve each single-shot result contract; do not invent universal correction equality across algorithms.
5. Optional reference environments must actually run with their dependencies installed. A missing-extra skip is not independent validation.

## Timing and resource protocol

- Reuse the maintained benchmark CLI/registry and `flagquantum.benchmarking.contract` metadata/writer when a new producer is justified. Reuse existing producers where they already express the measurement. Do not build another discovery or report system.
- Record exact source revision, tracked-tree cleanliness, package versions, platform, CPU model, thread counts, seeds, workload configuration, precision, and actual backend. Installed-artifact acceptance also records wheel/sdist hashes and import location.
- Freeze inputs before optimization. Suggested initial warmup/repetition counts are 3 and 10. Use paired/matched environments for comparisons. Cold construction/compilation and warm reused execution are distinct measurements.
- Timing boundaries include public-call overhead and all work claimed by the metric. Report model/schedule construction, fault/DEM construction, plan/compile, sampling, decoding, statistics, transfer and I/O separately where applicable. Aggregate an end-to-end wall time from an actual complete call, not a sum of incompatible best timings.
- Record raw observations and medians/uncertainty; do not publish only a speedup ratio. Retain refused, failed, timed-out and OOM cases in the matrix.
- Existing construction tracemalloc peaks exclude prior imports/layout construction. Process RSS is a whole-process high-water mark, not the numerical primitive's exclusive memory. Preserve those labels and units; do not compare them as equivalent metrics.
- Use fixed-batch streaming to study total-shot memory. Fresh subprocesses are needed for meaningful per-case process high-water memory comparisons. A reused-process RSS decrease does not establish lower peak memory.
- CPU/single-device observations remain local non-release evidence. Shot parallelism and genuine state/task sharding retain their existing distinct classifications. No distributed initialization is added to local reference examples.

### Proposed local resource envelope

| Item | Proposed bound | Interpretation |
| --- | --- | --- |
| Small reference/example command | 120 seconds. | Bounded development validation; a timeout is a failed/inconclusive run, not a performance result. |
| One construction point | 300 seconds initially, one CPU process/thread. | Stop and record timeout before trying a larger point; record allocated memory budget explicitly. |
| One source QEC acceptance selection | Existing bounded acceptance harness, 900 seconds. | Engineering verification, not a benchmark speed target. |
| Large shots / d>=9 / long history | Separate predeclared resource allocation. | Do not run an unbounded Cartesian product or silently omit large failures. |
| Hardware/experimental jobs | Named resource/data owner and approved execution scope. | No remote submission, paid job, credentials or new hardware execution is authorized by this document. |

These are proposed workload-execution limits for review, not relaxed CI limits. Existing stricter tests, timeouts and budgets remain unchanged.

## Numerical, statistical and performance decisions

| Decision | Proposed criterion | Approval status / restrictions |
| --- | --- | --- |
| Small exact reference | Complex128 probability/expectation absolute error <=1e-10, with normalization and discrete mapping checks. | Proposed narrow reference criterion. Do not apply to every representation or replace a stricter existing check. |
| Fixed-shot logical statistics | Existing shot-level failure definition and 95% Wilson interval; preserve per-observable counts. | Reuse delivered behavior; no redefinition and no denominator loss on failure. |
| Adaptive stopping / many comparisons | Predeclare a stopping-valid method and simultaneous comparison policy before a sweep. | Not approved or activated here. Repeatedly peeking at fixed-shot intervals is insufficient. |
| Throughput target | Proposed >=1.5x end-to-end benefit on two frozen workload classes, with uncertainty supporting improvement. | Not a current achievement or active CI threshold; approve before optimization. |
| Capacity target | Proposed >=2x primary executable scale dimension for an eligible class under the same resources and numerical semantics; retain separate throughput evidence. | Not a generic claim about arbitrary qubits/T count. Specify the dimension and failure baseline before measurement. |
| Domestic optimization | Same-device portable baseline, equal precision, full transfer/compile overhead included. | Device/stack and target classes must be selected; no undeclared host fallback. |
| Experimental prediction | Held-out detector/logic statistics and model discrepancy under a predeclared uncertainty procedure. | Dataset, mappings, sample sufficiency and validation limits must be approved. |

The system target in the master plan remains a proposal until the owner approves the selected matrix and budgets. No new performance gate is enabled by this PR. Baseline observations characterize a machine and revision; they do not approve the targets retrospectively.

## Runnable current-API example: non-Clifford interference and feedback

Core Torch is sufficient; no external engine, accelerator, credentials or dataset is required. Save as `ftqc_reference.py` and run `python ftqc_reference.py` against an installed distribution. Source users run `PYTHONPATH=. python ftqc_reference.py` from the checkout.

```python
import math

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

program = fq.Circuit(1).h(0).t(0).h(0)
options = fq.ExecutionOptions(
    mode="statevector", device="cpu", precision="complex128"
)
probabilities = fq.run(
    program, outputs=fq.probabilities([0]), options=options
).probabilities
assert probabilities.shape == (1, 2)
expected = torch.tensor(
    [(2 + math.sqrt(2)) / 4, (2 - math.sqrt(2)) / 4], dtype=torch.float64
)
assert float((probabilities[0] - expected).abs().max()) <= 1e-10
print("HTH probabilities:", *[f"{float(p):.8f}" for p in probabilities[0]])

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
result = fq.run(circuit, options=fq.ExecutionOptions(shots=128, seed=53))
assert result.samples is not None and result.samples.shape == (128, 2)
print("Feedback parity:", bool((result.samples[:, 0] == result.samples[:, 1]).all()))
```

Expected stdout:

```text
HTH probabilities: 0.85355339 0.14644661
Feedback parity: True
```

This demonstrates existing non-Clifford phase interference and software feedback. The parity check is an invariant, not a complete test of the Born distribution. The example does not demonstrate an encoded fault-tolerant protocol or a new mixed simulator.

## Reuse existing benchmark and sampling commands

After installing the project, the maintained CLI exposes current runners:

```bash
flagquantum-benchmark info dynamic_unified_run

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run dynamic_unified_run \
  --batched-shots 128 --trajectory-shots 8 --static-shots 128 \
  --warmup 1 --iterations 3 --planning-iterations 5 \
  --json-output /tmp/ftqc-dynamic-reference.json
```

The report includes measured planning/execution paths and the already approved wrapper budget verdict. Its actual timings vary. This short smoke is not the proposed 3-warmup/10-repetition comparative study and not a new performance acceptance decision.

Source users with the Stim/PyMatching extras can run the existing Surface workflow:

```bash
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py \
  --code surface --distance 3 --basis x --shots 128 --batch-size 32
```

Expected report invariants: `sampling="circuit"`, `code="RotatedSurfaceCode"`, `readout_basis="x"`, `requested_shots=shots=128`, `rounds=3`, `batch_size=32`, `stopping_rule="fixed_shots"`, and one logical observable. Failure counts and intervals are measured outputs, not cross-version pinned random values. Inspect `logical_support` and the declared noise configuration rather than assuming a different X readout construction.

The existing construction producer can be run from a clean source revision:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. python benchmarks/qec_clifford_dem_construction.py \
  --distance 3 --rounds 3 --output /tmp/ftqc-dem-reference.json
```

Its payload must continue to identify local construction evidence with scalability/release claims disabled. It includes private instrumentation internally; users need not import those helpers. Do not register a competing producer for the same construction operation.

## FTS-02 delivery and approval checklist

- [ ] Review the W1–W8 input matrix, available paths, independent references and explicit blockers.
- [ ] Approve the specific small-reference tolerance and bounded development envelope without changing existing stricter checks.
- [ ] Select the first frozen comparative workload classes, precision, hardware, resource budget, primary metric and repetition/statistical procedure.
- [ ] Record baseline observations from clean exact revisions through existing benchmark producers; retain raw artifacts and failures.
- [ ] Approve or revise proposed throughput/capacity commitments before optimizing against them.
- [ ] Name a domestic device/resource owner and an experimental dataset/partner for W7/W8; otherwise leave them blocked.
- [ ] Add only the missing bounded producer/conformance coverage justified by the approved matrix, reusing existing infrastructure.
- [ ] Verify user examples from the PR description at its final head and from a fresh installed artifact where applicable.
- [ ] Pass complete required CI, single-/two-GPU, review and conflict gates, then accept merged main before the next delivery.

Implementation of new APIs, approximation, FTQC package boundaries or hardware submission remains outside this proposal. DynamicCircuit #731, protected exports/identities, registration and all original frozen QEC fixtures remain unchanged.

## FTS-02E bounded scale campaign proposal

Status: **Stage A matrix and resources explicitly approved by the user on 2026-10-25 (Asia/Shanghai), with supervisor review required first**. This does not approve Stage B or waive the supervisor admission gate. The user's baseline-only decision still disables 1.5x/2x goals. No large run is performed by this proposal, no tolerance/CI threshold is changed, and no benchmark observation constitutes a capacity or performance improvement.

### Why this is the next decision

[W1-A phase observations and post-820 acceptance](FTQC_EXACT_REFERENCE_BASELINE.md) and [W6 phase observations](FTQC_SAMPLING_PHASE_BASELINE.md) are delivered. Existing `tests/team/simulation/test_stabilizer_sampling.py` already compares Clifford samples with dense probability references, including Bell/GHZ support; `test_prepared_stabilizer_sampling.py` independently authors Stim records and checks their joint distribution. These references should be reused during later W1-B/FTS-04 conformance work instead of adding a competing reference engine or claiming that parity alone verifies a Born distribution. Their mere presence does not close every W1-B fault/channel case or approve a new dynamic accuracy contract.

The first CPU construction study stopped at d=7; the maintained sampling observer intentionally admits at most 4096 shots. Expanding these ranges needs a declared resource budget and reviewed benchmark-envelope change. Both preserve existing public QEC APIs. The original complete-result decoder/experiment contracts remain unapproved; this campaign uses only existing native PyMatching logical predictions, not a hypothetical batch-result API.

### Stage A: approved pilot, supervisor review first

| Point | Frozen workload | Existing producer and proposed change | Records |
| --- | --- | --- | --- |
| A1 | Surface d=9, rounds=3, gate-bound `NoiseModel().add("cx", bit_flip_channel(0.01))`. | Existing `benchmarks/qec_clifford_dem_construction.py` unchanged. Construction only; no sampler/decoder/whole-circuit capacity conclusion. | One labelled host warm-up process, then three fresh measured processes. |
| A2 | Surface d=3, Z readout, rounds=3, data/phase/measurement flips 0.005/0.007/0.009, seed=53, 16384 shots, batch=32. | Existing `qec_memory_sampling` observer; a separately reviewed benchmark-only shot-envelope extension after approval. | One labelled host warm-up process, then three fresh measured processes, each `warmup=0`, `iterations=1`. |
| A3 | Same as A2, 65536 total shots, batch=32. | Same observer and explicit sample-count conservation. | Same four-process protocol; admit only after A2 completes within its resource budget. |

Host warm-up is not an in-process cache warm-up: all measured processes start fresh, and their first called-API lazy imports remain in measured phases. All host warm-up outputs remain in raw evidence. Three measured processes characterize a pilot; they do not satisfy a final performance comparison or infer a rare-event threshold. Seed restarts mean logical counts are not independent statistical trials. Report full observations/ranges without a speedup significance claim.

### Resource request and fail-closed admission

The request is one local CPU-only campaign on the existing host, one process at a time, one Torch/OMP/MKL/OpenBLAS thread, captured exact Python/Torch/Stim/PyMatching versions, fixed precision/record types matching the current producers, and no external hardware/job/data access.

| Resource | Proposed Stage A allocation | Handling |
| --- | --- | --- |
| Per A1 process wall | 300 seconds including startup. | Terminate owned process group and retain timeout evidence. |
| Per A2/A3 process wall | 120 seconds including startup. | Same; do not hide compilation/import overhead in the external duration. |
| Whole campaign wall | 45 minutes, including warm-ups and controller work. | Stop and mark remaining cases not attempted when exhausted. |
| Observed workload-process RSS | 2 GiB stop threshold, observed at <=100 ms intervals. | Requires functioning process supervision before launch. On a sampled exceedance terminate owned participants; retain high-water result where available, sample maximum, interval and termination latency. This sampled threshold can overshoot and is not a hard allocator/cgroup limit. |
| Retained evidence storage | At most 128 MiB; require at least 512 MiB filesystem free before launch. | Keep compact JSON/counts/logs, not all shot tensors. Refuse to start when space is insufficient; never delete unrelated files or original evidence. |

Reuse the maintained watchdog for wall-time/process-group cancellation. Its default hardware invocations remain wall-time-only. The optional RSS implementation described below must pass review and mainline acceptance before expanded collection. If RSS supervision is unavailable or permission is denied, record that blocker and do not start the expanded run. Do not substitute virtual-address limits for measured RSS. Parent/controller RSS, child high-water RSS, Python-traced allocations and visible tensors remain different measurements; none is renamed exclusive sampler memory. This proposal does not guarantee protection against an OS-level OOM.

For every point retain command, clean obtainable producer revision, workload/noise, environment, measured phases and actual external wall, process memory semantics, raw counts, normal completion/refusal/timeout/termination status, and logs. A failure cannot enter the successful median or disappear from the manifest. On a failure stop escalation in that workload class; keep other classes unattempted until the failure is reviewed. No automatic reruns of failed points and no selecting the fastest surviving subset.

The construction producer's traced peak excludes code/layout/import allocation and includes tracemalloc overhead in its construction timer. Preserve that distinction. Sampling's whole-process RSS includes imports and SDK preparation; visible detector/observable/prediction bytes do not establish a process-memory bound. Fixed batch and detector width are held constant for A2/A3; any memory change must be explained, without inventing an approved growth tolerance.

### Stage B: explicitly outside the pilot request

Distance 17, longer rounds, shots 10^6, batch-size comparisons, ten-repetition comparative studies and all new physical devices require a follow-up admission decision using Stage A observations. Stage A approval does not authorize Stage B or establish its budgets. W4 mixed-algorithm choice, approximate methods, ftqc package admission, public serialization/batch semantics, domestic device execution and experimental data remain separate contracts/decisions.

### Mission checklist and acceptance

- [x] Baseline evidence contributing to R01/R03/R08/R10, without closing these requirements: delivered small-code, sampling and exact-reference observations with phase/source/environment boundaries and accepted main installation.
- [ ] R01/R08: run the approved A1/A2/A3 pilot and preserve all failures and unattempted cases; this still does not close whole-circuit scale or streaming quality.
- [ ] R09/R10: select a future equivalence-preserving optimization and quantitative goals only after baseline review. No 1.5x/2x gate is active here.
- [ ] R04: mixed Clifford/non-Clifford algorithm selection remains open; H-T-H is a small reference only.
- [ ] R05/R06: reuse delivered feedback and independent small references, then close uncovered conditional/channel cases under authoritative contracts.
- [ ] R11/R12/R15: name domestic resource and experimental data owners; no resource or dataset is fabricated.
- [x] Explicit user approval of **Stage A matrix, repetitions, wall/RSS/storage allocations and supervision prerequisite** received on 2026-10-25 (Asia/Shanghai). This does not approve performance targets, new QEC APIs or Stage B.
- [ ] Complete supervisor review and mainline acceptance before extending the benchmark envelope or collecting Stage A cases.

Every implementation/evidence PR must continue to include a real public-user code example, prerequisites, command, expected output/invariants and exact reviewed-head execution evidence. The current H-T-H/feedback example above and the accepted 17-shot compiled-memory example remain runnable without activating this proposal.

### Stage A admission implementation: optional RSS watchdog

`tools/run_with_watchdog.py --rss-limit-bytes 2147483648` adds opt-in POSIX process-group RSS supervision to the existing wall-time watchdog. Existing hardware invocations omit the flag and do not query RSS. The supported workload is a trusted local command whose children remain in its owned process group; this is not isolation against untrusted commands escaping with `setsid`, an allocator limit, or a cgroup boundary. Shared pages may be counted once per group member. Short-lived allocations or processes between observations can be missed.

The monitor queries `ps` with a 50 ms query timeout, checks actual observation gaps against 100 ms, and stops on an unavailable/malformed query or excessive gap. An unavailable initial query prevents command launch (exit 125). A live RSS exceedance, lost accounting, excessive sampling gap, no live positive-RSS observation, or children outliving the launcher stops the owned group (exit 124). Otherwise the command's exit code is preserved. SIGTERM and Python interruption run owned-group cleanup; SIGKILL of the supervisor cannot run Python cleanup. Reuse parent/campaign supervision for that external failure boundary.

Reports retain sampled aggregate peak RSS, positive-RSS observation count, maximum actual interval, the last 32 timestamp/RSS pairs (a bounded recent trend, not every RSS sample), process return code and measured termination duration. Normal RSS-supervised completion writes `watchdog.json`; failures retain existing diagnostics. OS high-water RSS from producers and the supervisor's sampled process-group peak are distinct. All timing/memory fields are observed, not success thresholds for the numerical algorithms. A sampling-gap refusal never counts as an accepted resource observation. This change does not expand the current 4096-shot observer envelope or run the Stage A matrix.

Runnable public-user example (core CPU installation; source checkout for the watchdog). Save as `/tmp/ftqc_rss_example.py`:

```python
import flagquantum as fq

circuit = fq.Circuit(1).h(0).t(0).h(0)
options = fq.ExecutionOptions(mode="statevector", device="cpu", precision="complex128")
plan = fq.plan(circuit, outputs=fq.probabilities([0]), options=options)
probabilities = fq.run(plan).probabilities
print("HTH probabilities:", *[f"{float(p):.8f}" for p in probabilities[0]])
```

Run from the source checkout in an environment permitting read-only POSIX `ps` accounting:

```bash
PYTHONPATH=. python tools/run_with_watchdog.py --phase ftqc-reference \
  --stall-seconds 10 --timeout-seconds 120 --rss-limit-bytes 2147483648 \
  --diagnostics /tmp/ftqc-rss-reference -- python /tmp/ftqc_rss_example.py
```

Expected application output line: `HTH probabilities: 0.85355339 0.14644661`. The existing watchdog merges command stderr into its output, so dependency warnings may also appear and are retained. Exit zero requires the example to succeed and RSS monitoring to remain within the declared observation budget. Inspect `watchdog.json` for `reason="completed"`, `command_returncode=0`, a positive observation count, sampled peak no greater than the configured limit, actual maximum gap at most 0.1 seconds and `hard_allocation_limit=false`. Timings/RSS values depend on the host and are not pinned. This small example is not the approved larger campaign.
