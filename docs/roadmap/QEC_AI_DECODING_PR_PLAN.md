# AI quantum-error-correction decoder delivery plan

Planning baseline: 2026-10-24. This document records delivery gates, not supported
capabilities or performance promises. Existing public APIs remain unchanged.

## Current state

### Progress snapshot: 2026-10-26 (Asia/Shanghai)

**The matched CNN/MLP study is not ready for formal execution.** Its scientific
implementation and several private controller components have passed software
verification, but the final staging/collection path and actual trusted caller
remain incomplete. No new matched-study formal training, physical-test sampling
or decoder-quality result is established by these controller checks. The
optimization, export, domestic-accelerator and streaming gates below remain closed.

| Area | Completed and evidenced | Still pending |
| --- | --- | --- |
| Matched-study implementation | PRs #847, #849, #850, #864 and [#871](https://github.com/flagos-ai/FlagQuantum/pull/871) are merged. The fixed training, calibration, validation, probe, primary replay and summary workflow is implemented and tested. | Executing the new frozen formal study and evaluating its scientific gates; implementation is not a quality result. |
| Scientific source | The retained producer is `fc07f4dc7cdb92068460afcee629930791800345`, with source freeze, dependency copy, input binding and source delivery completed. | Do not substitute a later `main` revision or claim tree equivalence. A changed scientific producer needs its own final CI and fresh freeze. |
| Actual two-host preflight | Both A800 hosts passed zero-sampling environment, layout and model-loading checks with distinct GPU UUIDs. Both owned containers were stopped and removed. | A current live formal scope, authenticated caller/ownership/UUID binding and study prediction-parity receipts. Historical removed containers are not current readiness evidence. |
| Private final-consumer chain | Optional native validation-parent → issuer → read-only final consumer is connected using the same caller Budget and original pre-freeze process end. Independent receipts, byte capture and failure checks are implemented. | This is private software integration, not authenticated host authority or permission to stage/transport final artifacts. |
| Stage and transport | Existing capture, partition, independent shard copy, receive/assembly and collection primitives are retained. Stage contract review, optional stage Budget forwarding and independent read-only consumer parent-complete acceptance are implemented and software-tested. | The final-consumer-to-stage adapter and whole-stage owned-worker deadline/tamper acceptance are not wired. Final-purpose collection remains disabled; the existing prepared-purpose route is not relabeled. |
| Formal launch | The recipe, evidence requirements and failure rules remain frozen. | Full prepare/ready/peer-receipt/finish/failure/collection acceptance, authenticated live controls and fixed-recipe scientific storage preflight. Observed free space is not a reservation or a certified storage budget. |

The private stage audit identified missing Budget routes. The subsequent bounded
software slice adds keyword-only `budget=None` to `artifact_stage.stage` and
`_copy`, forwarding the same object to the original copy, verify and both source
capture operations. Entry and post-complete checks retain 136 seconds of Budget
reserve. Old defaults, Snapshot/Plan timestamps, receipt schemas and all original
storage reserves remain unchanged; no new stage arm or outer deadline is created.
Only the existing issuer's stage-source pin changes alongside those two functions.

The next bounded slice adds independent read-only consumer parent-complete
acceptance in the existing consumer runner. It checks the external completion and
request hashes, reconstructs the exact 39-file table and all five external seals,
and reuses existing worker receipt/ancestor validation. The original candidate,
Snapshot bytes/digest, current payload inventory/stat seal, original source and
fresh read-only source copy must agree. A self-reported table is not accepted as
the expected inventory. No copying, writing, arm, deadline remapping, payload
capture or `.pt` read occurs in this entry.

The caller must supply its existing local Budget and exact original pre-freeze
end in the same clock domain; the recorded end is compared, not converted into a
new deadline. Full prior costs/reserves, actual runner-source bytes and the old
control allowance remain charged without retained-byte credit. Metadata/source/
space checks finish with current payload-state and deadline checks. The state
check reuses the original captured stat seal; it is not a new full payload-byte
reread, atomic reservation or independently authenticated host authority.

These entries are not yet connected to a stage adapter or owned stage worker.
Full post-complete and post-callback ancestry, payload/copied-byte, inventory,
source and space tamper acceptance remains pending for that future composition.
Blocking parent OS I/O remains synchronous. Final-purpose collection stays
disabled; software acceptance does not establish formal readiness.

### Next bounded work

1. Reuse the completed independent consumer parent-complete acceptance before
   any copying in the minimal explicit final-consumer-to-stage adapter. Keep its
   external hashes, exact 39-file table, all five ancestor seals, original Snapshot
   and current stat/source/fresh-copy checks. A returned Snapshot alone is not
   permission to stage or transport. Preserve old defaults, prepared-purpose
   behavior and raw schemas; do not reimplement the existing receipt authorities.
2. Keep the same local caller Budget and exact original pre-freeze outer end.
   Restore original controls and arm once locally in an owned worker; do not
   serialize a Budget, renew either original deadline or remap a later outer end.
   Before freezing, charge all retained prior costs/reserves plus actual new
   source, requests, bounded logs, metadata, independent copied shards and later
   retained wire/receive/assembly copies. Merge charges on shared filesystems;
   do not credit old retained bytes or treat transport limits as a science budget.
3. Verify all three deadline orderings, blocked copy/verify/source/metadata paths,
   owned-group termination, permanent clock-failure poisoning and late tampering.
   Check again after complete writes and native callbacks; failure must retain
   invalidated owned partial evidence and must not return success or allow retry.
   Keep final-purpose collection disabled until this path is fully accepted.
4. Then complete the actual canonical-source/strict-SSH/owned/live-UUID caller,
   full phase/failure/collection acceptance and fixed-recipe storage gates before
   considering a formal launch. Software passes alone authorize neither actual
   transfer nor formal optimizer/sampling/GPU jobs. Retain frozen and failed
   evidence; do not restart a formal partial, replace seeds or expand its budget.

### Evidence and verification boundaries

The frozen scientific recipe and layout correction are retained in
[matched comparison](../development/QEC_AI_ISING_CNN_COMPARISON.md) and
[layout verification](../development/QEC_AI_ISING_CNN_LAYOUT_VERIFICATION.md).
The [earlier continued physical outcome](../development/QEC_AI_CONTINUED_PHYSICAL_ASSESSMENT_RESULTS.md)
remains inconclusive and is not replaced by the new workflow's software tests.
The frozen producer's [CPU CI](https://github.com/flagos-ai/FlagQuantum/actions/runs/40194297369)
completed all 54 jobs; its [GPU CI](https://github.com/flagos-ai/FlagQuantum/actions/runs/40194297211)
completed with three successful and three skipped jobs. These certify that tested
revision's software, not a later `main`, private controller code or formal readiness.

Private evidence below is retained **outside this checkout**, in the sibling
`qec-ai-ci-evidence` directory. The paths below are local record identifiers, not
repository hyperlinks or published artifacts available in a standalone clone.
Their hashes identify the reviewed records; they do not turn private tests into
repository CI. The handoff index is a living local record rather than an immutable
published evidence bundle.

- Actual two-host preflight, local record
  `qec-ai-ci-evidence/matched-private-actual-preflight.json`, SHA-256
  `d4f7c3149d61c1e3601c44603c1ef814f8a38805b2c96977e61b38e33a943fd8`:
  zero new samples/optimizer updates; environment/layout/model loading only,
  without study CPU/CUDA prediction-parity or scientific-quality certification.
- Native consumer integration, local record
  `qec-ai-ci-evidence/matched-private-native-consumer.json`, SHA-256
  `32a107f93acbede9a04dab933e7515fce727d34967d9753adcb8b1b21e038b9e`:
  final new matrix **45 passed in 108.99 s**; separate final regression selection
  **60 passed, 83 deselected in 21.41 s**. Strict mypy checked 81 files; Ruff
  0.16.0 and Black 26.10.1 checks passed. Tiny study data and host/ownership/capacity
  boundaries are synthetic; this is not a full-private-suite or actual-host pass.
- Read-only stage contract audit, local record
  `qec-ai-ci-evidence/matched-final-stage-contract-audit.json`,
  SHA-256 `5bac2bd7cd95ed9abfbaaceec9493e7269a7645b21644ec790d32c86ee1424b8`:
  records source/function hashes, exact existing bindings, missing Budget routes,
  storage obligations and acceptance requirements. Existing pure planner tests
  were **34 passed, 1 deselected in 0.03 s**; zero new cases, no adapter or production
  changes, and no stage/transfer execution. This is a static audit, not integration
  acceptance.
- Stage optional Budget-forwarding software, local record
  `qec-ai-ci-evidence/matched-private-stage-window.json`, SHA-256
  `de3c54d4b6c5b27a924a8cae340f27e22bca27073ed28fdf42e11518de1b6119`:
  **17 new cases**; final related selection **148 passed in 19.18 s**, including
  existing native/consumer three-ordering positives. Separately, **2 existing
  full-charge one-byte-short checks passed in 9.04 s**. Strict mypy checked 82 files;
  Ruff 0.16.0 and Black 26.10.1 checks passed. Original source/test history and the
  two-function/one-pin AST review are recorded in
  `qec-ai-ci-evidence/matched-stage-window-helper-change.json`.
  Tiny local copies, original codec and capture are real; clocks, lease ordering,
  capacity and host/approval boundaries are synthetic. The slice adds no retained
  files or fields and does not certify a future stage-worker aggregate charge,
  blocked-body termination, consumer-completion acceptance or final transport.
  These private results are not repository CI or a full-private-suite pass.
- Independent consumer parent-complete acceptance, local record
  `qec-ai-ci-evidence/matched-private-consumer-complete.json`, SHA-256
  `3bd9284a86d5f279f55a17697b79183312d1577b8885f68373910f15725ad321`:
  **36 new cases**; related selection **97 passed**, including the original window,
  native/consumer three-ordering positives and full-charge shortage checks.
  After a test-only direct-import typing correction, a focused regression selection
  passed again; exact counts and timings are retained in the record and logs.
  Strict mypy checked 83 files; Ruff 0.16.0 and Black 26.10.1 checks passed.
  Original runner function ASTs, defaults/return/raw/end and all other 81 existing
  source/test files and pins are unchanged. The record covers rehashed completion
  forgery, ancestor/current-state drift, late source/space/control changes,
  permanent clock poisoning and full-charge one-byte shortage. Tiny sealed
  source/codec/receipt checks are real; study, clocks, lease, capacity and
  host/approval/UUID boundaries are synthetic. No copy or formal operation was
  authorized or performed. This is not repository CI or a full-private-suite pass.
- Retained handoff index, local record
  `qec-ai-ci-evidence/pr871-matched-primary-handoff.json`:
  links per-slice evidence, logs, historical failures and immutable source/proof
  records. Test-file collection counts and earlier partial runs are not full-suite
  pass counts. The local source cache has read-only regular files and 115 writable
  directories; it must not be described as having read-only directories throughout.

### Earlier results (retained history)

The entries below describe earlier studies and delivery decisions. The dated
snapshot and next-work list above supersede their historical sequencing.

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

The user-authorized
[A800 checkpoint comparison](../development/QEC_AI_ISING_A800.md): d=9/R=9,
explicit FP32 GPU inference, CPU/GPU correction-bit parity before test sampling,
and synchronized end-to-end timing including transfers and CPU matching.
Native training stays on the existing CPU trainer; the official checkpoint is
not an equal-budget architecture control. The new lane measures the missing
external baseline; it does not advance the gated optimization, export or domestic
accelerator stages below.

Both full A800 host runs are now retained in the
[outcome report](../development/QEC_AI_ISING_A800_RESULTS.md). On 65536 shared
physical shots, global-only and native paths each failed 83 times; Fast failed
72 times (61 wins, 50 losses). Every paired gate is inconclusive and every
native threshold is 1.0. Global-only remains fastest in this unoptimized software
configuration. CUDA allocator peaks are valid; raw host RSS values are explicitly
quarantined because Linux workers inherited the parent's high-water mark. A
post-run reporting fix does not retroactively recover that measurement.

PR #831 adds exact correction-label equivalence, streaming MLP feature groups
and a verified archived-checkpoint runner. Its fixed
[diagnostic outcome](../development/QEC_AI_LABEL_IDENTIFIABILITY_RESULTS.md)
retains three 4096-shot independent-DEM streams: all four controls stayed
inactive, with negative raw logits despite some high observed group positive
fractions. This is descriptive training-distribution evidence, not a new logical
accuracy or performance result. Next freeze a training-sufficiency study for the
unchanged model before choosing a loss or feature intervention. The separately frozen
[training-response outcome](../development/QEC_AI_TRAINING_SUFFICIENCY_RESULTS.md)
retains all three continued trajectories at steps 200/600/2000: all three pass
the descriptive held-out BCE response rule, while only two activate at raw
threshold 0.5. This is not convergence or logical-quality evidence. Next freeze
fresh calibration/validation and physical-test precision budgets before any
logical-quality comparison; retain global-only and official Ising Fast controls. Ising is a mandatory external
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
container has no usable CUDA mapping and was not changed. Both A800 hosts subsequently completed the frozen d9 comparison in the same
image and dependency environment; see the linked outcome for UUIDs, hashes,
quality limitations and measurement boundaries.

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

The prerequisite PRs above have merged; the former instruction to stack PR 2 on
#770 is complete. Future dependent changes must still rerun checks against their
actual base before merge. Do not bundle later-stage implementations into a
prerequisite PR. Keep protected API/schema changes in separately reviewed contract
proposals before their consumers change.

## AI implementation organization

The directory consolidation following PR #786 places the existing implementation
under `flagquantum/qec/decoders/_ai/`: `predecoder.py`, `spacetime.py`,
`training.py` and `assessment.py`. Examples and behavioral tests use these
canonical private paths. The former flat private modules are removed without
forwarding compatibility layers. Public QEC exports, decoder registration and
training checkpoint contents remain unchanged. See the
[local contributor guide](../../flagquantum/qec/decoders/_ai/README.md).

The consolidation was a behavior-preserving prerequisite, not a new model or an
accuracy/performance result. Label/training-response diagnostics, calibration and
validation-only threshold-selection software have since been delivered; their
retained outcomes do not establish the new matched study's scientific gate.
Current pending work is the stage/controller acceptance listed in the dated
snapshot above. Preserve the original studies' inconclusive results and hashes.
Proceed to performance and hardware work only after the scientific gate is met.

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
