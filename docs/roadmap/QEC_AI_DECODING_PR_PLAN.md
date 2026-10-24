# AI quantum-error-correction decoder delivery plan

Planning baseline: 2026-10-24. This document records delivery gates, not supported
capabilities or performance promises. Existing public APIs remain unchanged.

## Current state

PRs #770, #775, #780 and #786 are merged: the private CPU chain now includes
independent training-data checks, a geometric 3D CNN and prospective paired
assessment. PR #786 merged at `046384975775879efc4db9f38403614b0f3a35a1`
after complete CI and the repaired two-GPU warmup gate passed. This is software
verification, not evidence that the AI decoder improves accuracy or latency.

The first frozen study retains all 54 seed/variant/cell groups at 16384 shots
per group. Every selected threshold was 1.0, hybrid and baseline logical failure
counts were equal, and every quality decision was inconclusive. No scientific
gate or hardware capability is promoted.

PR #795 merged the private-package consolidation; the flat paths are retired.
PR #801 merged diagnostics, calibration and validation-only threshold selection.
Its fixed-budget study improved probability scores descriptively, but every
selected threshold remained 1.0 and every paired LER gate remained inconclusive.
See the retained [calibration outcome](../development/QEC_AI_CALIBRATION_RESULTS.md).

PR #812 is merged with complete CI: full circuit-level hyperedges, shared batched
global decoding and timing, and upstream input/frame checks are implemented.
The [CPU protocol](../development/QEC_AI_ISING_COMPARISON.md) and
[outcome](../development/QEC_AI_ISING_COMPARISON_RESULTS.md) retain the initial
d=3 exploratory comparison and its disjoint-noise approximation. Official Fast
weights are now authorized and hash-verified; missing access still fails closed.
Native thresholds remained 1.0 and all paired quality summaries were inconclusive.

The next independent slice is the user-authorized
[A800 checkpoint comparison](../development/QEC_AI_ISING_A800.md): d=9/R=9,
explicit FP32 GPU inference, CPU/GPU correction-bit parity before test sampling,
and synchronized end-to-end timing including transfers and CPU matching.
Native training stays on the existing CPU trainer; the official checkpoint is
not an equal-budget architecture control. The new lane measures the missing
external baseline; it does not advance the gated optimization, export or domestic
accelerator stages below.

After those prerequisites, diagnose label equivalence and training sufficiency
before selecting a single scientific intervention. Ising is a mandatory external
baseline alongside global-only decoding and native BCE controls. Compare common
circuits/shots, global-decoder settings, hardware and precision; report paired LER,
full software latency, throughput and memory. Keep equal-budget training studies
separate from comparisons with external pretrained checkpoints, and common CPU
execution separate from independently optimized accelerator deployment. Smoke
and single-cell exploratory results do not advance optimization/hardware gates.

GPU research runs are authorized on the user's A800 hosts `a800-172` and
`a800-171`; the local workstation has no GPU and remains a CPU correctness host.
Read-only inspection on 2026-10-24 found eight A800-SXM4-80GB devices per host.
This is availability inspection, not a reservation or benchmark. Preserve other
sessions, use an isolated job directory, declare a device/precision/shot/timing
protocol before experiments, and compare contenders on the same host/device.
The existing frozen CPU protocol is not silently changed into a GPU study.
Validate the official model's stated distance range (Fast: d >= 9) before
making a representative comparison; the current d=3 cell remains exploratory.
An isolated, auto-removed container on A800-172 successfully ran a four-element
CUDA tensor check with PyTorch 2.13.0+cu130. The existing shared `fq-it-gpu`
container has no usable CUDA mapping and was not changed. A800-171 has NVIDIA
container runtime configured but does not yet have the same FlagQuantum image;
prepare an isolated matching environment before cross-host replication.

## Delivery sequence

| PR | Deliverable and reuse | Acceptance gate | Dependency |
| --- | --- | --- | --- |
| 1 / #770 | Reference chain; reuse DEM, memory construction and decoder factory | Changes pass CI; training/resume/evaluation reproducible; record unrelated failures separately | Existing QEC restructure |
| 2 | Independently validate training data; reuse code records and circuit sampling | X/Z memories, multiple rounds/rates; latent label parity; detector/logical joint statistics against independent circuit execution; reproducible separated random streams | #770 |
| 3 | Geometric input layout and local 3D CNN; derive detector identity from existing layouts | Boundary masks, mechanism mapping and Hc/Lc tests; train/resume/evaluate surface memories; retain MLP comparison | PR 2 |
| 4 | Scientific evaluation and M2 decision | Predeclare train/validation/test grid, at least 3 training seeds, paired statistics, confidence intervals, ablations, noise/distance/long-history generalization | PR 3 |
| 5 | Profile-driven single-machine optimization | Exact discrete-operation parity; measured full software latency and LER, batch=1 and throughput modes, declared timing boundaries | PR 4 decision |
| 6 | Independent model export/load | Bind weights, DEM, layout, threshold, version and support scope; reject mismatches; verify exported quality | PR 5 |
| 7 | One selected domestic accelerator | Device evidence, numerical/quality equivalence and end-to-end timings; no undeclared CPU fallback; quantization separately gated | PR 6 |
| 8 | Streaming window execution | No lost/duplicate commits; boundary/reset recovery; bounded memory; sustained replay without queue growth; tail latency includes window wait | PR 7 |
| Later | Real-QPU closed loop | Device-specific measurement/control contract, deadlines, experiment data and sustained feedback evidence | PR 8 plus hardware team |

PR 2 may be prepared as a draft stacked on #770. Retarget it to main after #770
merges and rerun checks before merge. Do not bundle later-stage implementations
into a prerequisite PR. Keep protected API/schema changes in separately reviewed
contract proposals before their consumers change.

## AI implementation organization

The directory consolidation following PR #786 places the existing implementation
under `flagquantum/qec/decoders/_ai/`: `predecoder.py`, `spacetime.py`,
`training.py` and `assessment.py`. Examples and behavioral tests use these
canonical private paths. The former flat private modules are removed without
forwarding compatibility layers. Public QEC exports, decoder registration and
training checkpoint contents remain unchanged. See the
[local contributor guide](../../flagquantum/qec/decoders/_ai/README.md).

This is a behavior-preserving prerequisite to the next scientific improvement,
not a new model or an accuracy/performance result. Next diagnose label/loss
imbalance, confidence calibration and validation-only threshold selection inside
the same package, with a new frozen experimental budget before collecting test
results. Preserve the original study's inconclusive results and hashes. Proceed
to performance and hardware work only after the scientific gate above is met.

## PR 2 verification design

The native rotated-surface record declares logical Z only. For X/Z comparison,
use its CSS check matrices and explicitly declare the complementary column-X
operator through the existing validated `CssCode` constructor; no native X-memory
support is implied.

1. Recompute each sample's detector and logical bits by XORing the *recorded
   mechanisms' original signatures*, without using the learner's H/L buffers.
   This establishes latent-label integrity, not independent physical correctness.
2. Execute the same X/Z surface memory on the existing Stim-backed circuit
   sampler. It lowers and executes positioned faults rather than using the
   learner's DEM-matrix sampler. Compare detector/logical marginal and parity
   statistics with a declared family-wise concentration bound. Do not compare
   shot-for-shot samples from different random engines.
3. Treat circuit/model/layout construction as shared input: this comparison
   independently checks the new sampler, not all QEC semantics. Retain the
   existing independent circuit-transcription tests as the builder reference.
4. Recreate training/validation/test generators with distinct explicit seeds;
   verify reproducibility and that held-out evaluation does not advance the
   training generator. Independent seeds do not imply disjoint physical errors.
5. Optional Stim tests must run in an environment with Stim installed. A skipped
   reference test does not satisfy the independent-validation gate.

## Scientific and systems gates

Before PR 4 experiments, select a primary objective on a frozen test matrix:
LER reduction of at least 10% with statistical evidence, or full software speedup
of at least 1.5x with a one-sided 95% upper confidence bound on the LER ratio no
larger than 1.05. These are proposed targets, not measured outcomes. Test budget
exhaustion with insufficient precision is inconclusive. Report all training seeds
and zero-failure upper bounds. Threshold selection uses validation data only.

If PR 4 establishes neither quality benefit nor credible speed potential, return
to model/loss design before accelerator work. If PR 5 gains disappear after
transfer, conversion and global decoding, optimize the measured bottleneck before
increasing model size. If PR 8 cannot sustain the specified arrival rate, retain
the offline product scope until the queueing problem is resolved.

## Review and evidence

Each PR must include a runnable user workflow, support/refusal scope, focused
behavioral tests, relevant API/compatibility checks and exact validation results.
Use `python tools/ci_tier.py pr-default` plus the relevant integration tier;
report timeouts, skips and baseline failures without relabeling them as passes.
Do not lower CI gates or alter stable exports to accommodate an experimental
implementation. Changes and conclusions must name the tested commit/environment.

Implementation detail and reference commands for PR 1 remain in
[QEC AI M0/M1](../development/QEC_AI_M0_M1.md). New data checks are in
`tests/qec/test_neural_training_data.py` and can be run with:

```bash
python -m pytest tests/qec/test_neural_training_data.py -q
```
