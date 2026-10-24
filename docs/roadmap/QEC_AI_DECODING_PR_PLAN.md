# AI quantum-error-correction decoder delivery plan

Planning baseline: 2026-10-24. This document records delivery gates, not supported
capabilities or performance promises. Existing public APIs remain unchanged.

## Current state

PR #770 (`https://github.com/flagos-ai/FlagQuantum/pull/770`) establishes a private
CPU reference for sampling, local neural scoring, training, checkpoint/resume,
residual/global decoding and held-out evaluation. Its model is a shared local MLP,
not the planned geometric 3D CNN. It demonstrates reproducibility, not accuracy
or latency improvement. Its CI fixes were pushed at `2e2ac67`; full CI and review
remain the merge gate. No automatic merge is authorized by this plan.

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
