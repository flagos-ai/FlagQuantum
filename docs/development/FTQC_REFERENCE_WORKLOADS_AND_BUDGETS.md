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
