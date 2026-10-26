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
| Stage and transport | Existing capture, partition, independent shard copy, receive/assembly and collection primitives are retained. Stage contract review, optional stage Budget forwarding, independent consumer parent-complete acceptance and an explicit synchronous consumer-to-stage adapter and sealed stage worker entry are software-tested. The owned-session harness covers local restore/arm and blocked-body failures; an independent read-only parent acceptance API is now tested. | Production stage outer runner/parent completion, native stage handoff and full post-native-callback closure remain unwired. Final-purpose collection remains disabled; the existing prepared-purpose route is not relabeled. |
| Formal launch | The recipe, evidence requirements and failure rules remain frozen. | Full prepare/ready/peer-receipt/finish/failure/collection acceptance, authenticated live controls and fixed-recipe scientific storage preflight. Observed free space is not a reservation or a certified storage budget. |

The private stage audit identified missing Budget routes. The subsequent bounded
software slice adds keyword-only `budget=None` to `artifact_stage.stage` and
`_copy`, forwarding the same object to the original copy, verify and both source
capture operations. Entry and post-complete checks retain 136 seconds of Budget
reserve. Old defaults, Snapshot/Plan timestamps, receipt schemas and all original
storage reserves remain unchanged; no new stage arm or outer deadline is created.
Only the existing issuer's stage-source pin changes alongside those two functions.

The retained read-only entry provides independent consumer parent-complete
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

The explicit synchronous adapter now calls that independent entry before any
output creation or copying, then reuses the original stage/capture/copy/codec
path with the same Budget. It derives the source from the original consumer
ancestry, validates the supplied Plan and keeps its original timestamps. A fresh
owned wrapper retains the adapter source, lock, unchanged stage subtree and a
completion linking the external consumer/request/Plan/Snapshot and stage hashes.
Both returned hashes are software evidence, not transport or launch permission.

Before output creation, the adapter conservatively co-locates the full original
consumer/native/issuer charge, stage sender footprint, existing collection-window
receiver/controller/wire/log/request/observer/assembly costs and the new adapter
source/control allowance. Each prior/new output parent must pass the whole
aggregate, without retained-byte credit or subtraction of any original reserve.
This is a local preflight, not remote authentication, a reservation or a certified
scientific storage budget. The new sealed worker additionally charges its actual
32-file source snapshot, two worst-case 528 MiB logs, two exact requests, two
16 MiB metadata allowances and 1 MiB of controls, retaining every old reserve.

Independent copied-byte verification and repeated external consumer acceptance
run after stage and wrapper completion. A fault regression exposed an initially
rebased copy-state baseline: equal-length mutation at the last codec return could
escape. The correction fixes the stage inventory/directory stat baseline before
independent verification and compares that same baseline before/after each check.
The original failure log and source are retained; original codec checks were not
weakened. Owned late failures retain candidate/completion plus failure evidence
and cannot resume; concurrently created foreign output is left untouched.

The new sealed stage entry restores the original window, Plan, time approval and
supplied owned controls through the existing issuer/validation request reader,
then arms once locally. It receives an externally hashed nine-field request,
never a serialized Budget. The original consumer completion/request hashes, five
ancestor seals and Plan hash remain bound. The same child-local Budget and exact
original pre-freeze end reach the completed adapter; original raw deadlines,
stage defaults and receipt purposes remain unchanged. The fixed bootstrap checks
32 actual fresh source files before import. The adapter's full-charge formula is
mechanically extracted for reuse; all other 84 prior source/test files and pins
remain unchanged.

A real owned-session harness reuses the original bounded process/group helper.
It checks the full aggregate at every original/new output parent before freezing;
the worker checks again before output creation and after completion. Source,
metadata, copy and verify pipe blocking is stopped with owned-group termination,
including descendants that ignore TERM. Only the test timeout invocation is
compressed to 1.5 seconds; the request keeps the original end and raw controls.
Three original deadline orderings, one local arm, clock poisoning, copied bytes,
peer, fresh source, retained adapter source, result, extra inventory and late
space failures are covered. Completed failed roots and logs are retained.

The subsequent read-only parent API independently accepts the externally hashed
worker completion/request and bounded stdout using the caller's existing Budget
and original end. It reconstructs the worker's 32-source lock, result, exact
completion table and false authorization flags; rehashed empty tables are not
accepted as expected inventories. Existing independent consumer acceptance
rechecks the 39-file table, five ancestor seals, current Snapshot/stat state and
original/fresh sources. The original partitioner, stage binding and codec remain
the Plan, receipt and copied-byte authorities.

The API reconstructs the adapter's 31-source lock and completion/manifest table,
binds the original stage lock's source and capacity metadata, verifies actual
copied bytes without aliasing, and checks the exact read-only fresh 32-file source
snapshot. One fixed inventory/directory stat baseline encloses verification and
all final source/space/ancestor reads; it is not rebased after codec return.
The whole existing aggregate plus actual acceptance-source bytes and a 1 MiB
control allowance is checked at every prior/new output parent without retained
or reserve credit. This read-only entry creates no new retained files or logs;
future runner costs still need explicit charging.

The API does not arm, remap time, capture, copy, stage, write approval or rewrite
producer completion/failure evidence. A future owned outer runner must turn any
refusal into its own failed, non-retryable transaction. Parent codec/source/hash/
metadata/stat I/O is synchronous and filesystem checks remain non-atomic.
Production stage outer runner/parent completion, native handoff/full callback
closure and final-purpose collection remain unwired. The existing per-chunk
Budget/codec mapping stays unchanged; software passing is not formal readiness.

### Next bounded work

1. Reuse the completed independent stage acceptance API, sealed entry, adapter
   and original process helpers. Add the smallest production outer runner with
   externally bound request/result/completion, fresh actual source snapshots and
   a parent completion seal. Use the existing independent reconstruction and
   copied-byte/current-ancestor checks; do not introduce another receipt, Plan,
   decoder or statistics authority. Keep defaults/raw schemas/prepared purpose.
   Retain failed owned roots and refuse retries, including after parent completion
   and any later native callback. A returned hash alone is not permission.
2. Keep the same local caller Budget and exact original pre-freeze outer end.
   Reuse the worker's original-control restore and single local arm; do not
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
- Synchronous consumer-to-stage adapter, local record
  `qec-ai-ci-evidence/matched-private-stage-adapter.json`, SHA-256
  `92b785280873d8e55ffc22af3e9626f958cf31cccccba13325e49d434eba5597`:
  **28 new cases**; final related selection **101 passed in 111.77 s**, including
  existing stage/window and three-ordering consumer acceptance regressions.
  Strict mypy checked 85 files; Ruff 0.16.0 and Black 26.10.1 checks passed.
  All 83 prior source/test files and pins remain unchanged. The initial final-codec
  mutation regression was **1 failed, 1 passed in 9.49 s**; after the fixed-state
  correction, **2 passed in 9.32 s**, and both are included in the final selection.
  Evidence retains the failed source/history and logs, conservative full-charge
  shortage, before-copy refusal, copy/verify failure, late tamper, clock poison,
  retry refusal and foreign-output checks. Tiny sealed codec/replay/capture/copy
  operations are real; study/host/UUID/lease/approval/capacity/clocks are synthetic.
  Its historical completion record observed 925696000 local free bytes, below
  the unchanged 1 GiB reserve. That observation is not a current capacity claim;
  no evidence was deleted or reserve reduced to bypass the gate.
  This is not a blocked-stage-worker, native stage integration, repository CI,
  full-private-suite, actual transfer or formal-readiness certification.
- Sealed stage worker entry and owned-session harness, local record
  `qec-ai-ci-evidence/matched-private-stage-worker.json`, SHA-256
  `87a064014fb04210c7411874cb4bdeb91fe6c39cbbff64b81d064d7f71d8ac8e`:
  **8 new pytest cases**, with **57 worker scenario executions** inside their
  matrices; final related selection **67 passed in 105.71 s**. Strict mypy checked
  87 files; Ruff 0.16.0 and Black 26.10.1 passed. Three deadline orderings, four
  blocked paths, entry/post-complete clock poisoning, late tampering and full
  worker charge one-byte shortage are covered. The initial final matrix had
  **1 failed, 66 passed in 107.64 s** from recursive test patching; only the test
  alias was corrected. Earlier fixture-capacity/exception failures and a wrong-cwd
  no-test invocation remain in history. No production guard was weakened.
  The adapter charge-extraction audit is local record
  `qec-ai-ci-evidence/matched-stage-worker-helper-change.json`, SHA-256
  `008258d2ff894668cafdbbd8a9f33d63882e3e78181f5d9082d67a8a1ff8cebc`.
  All four actual proofs and the frozen scientific producer remain unchanged.
  Completion observed 1216512000 local free bytes; the complete aggregate still
  cannot fit. Observed space is not a reservation or scientific storage budget.
  Tiny original codec/replay/parser/capture/copy and owned process termination are
  real; study/host/UUID/lease/approval/clocks/capacity success remain synthetic.
  No actual SSH, resource probe, GPU/container, preflight, transfer, study approval,
  formal sample, update or receipt was performed. This is not repository CI, a
  full-private-suite pass, production parent acceptance or formal readiness.
- Documentation PR #895 was merged by the user at 2026-10-26 06:23:57 UTC.
  Exact head `feba3c70b44f2ca33db2fae412a57412cb6aed96` completed CPU run
  `40277743274` with 54 successful jobs and LocalGPU run `40277743260` with
  three successful and three skipped jobs. Local record
  `qec-ai-ci-evidence/pr895-user-merge-final-ci.json` retains the job-level check.
  This does not certify private worker code or subsequent main revisions; the
  agent performed no merge and the retired documentation branch is not reused.
- Independent read-only stage-worker parent acceptance, local record
  `qec-ai-ci-evidence/matched-private-stage-parent.json`, SHA-256
  `de22033785a7137fcc498551085f24f964baa7bff667068ba8e0791ab3ba7dc9`:
  **9 new pytest cases**; final related selection **79 passed in 20.19 s**,
  including 53 existing window and 17 stage Budget cases. Final testing reused
  three retained tiny candidates with zero new worker invocations. Earlier fresh
  runs were 3 passed in 28.60 s, then 9 passed in 79.58 s before added frontend and
  shortage checks. Strict mypy checked 89 files; Ruff 0.16.0 and Black 26.10.1
  passed. All 87 prior source/test files and pins remain unchanged; integrity
  local record `qec-ai-ci-evidence/matched-stage-parent-integrity.json`, SHA-256
  `04f5de8117d657d45e9d000a2e3586be58b892db138eae3c38557c736a1a7086`.
  Three-ordering checks cover 51 rehashed/late tamper refusals, 30 entry/after-codec
  permanent clock failures, nine frontend refusals and three complete-aggregate
  one-byte shortage refusals. Success forbids arm/capture/copy/stage/write and
  preserves producer inventory. Rejected synthetic extras are retained in sibling
  records; actual, frozen, failed and scientific evidence is not deleted.
  An intermediate selection was **6 failed, 76 passed in 97.21 s**: three new
  shortage tests still called a fixture substitute, and three old adapter
  positives correctly refused actual low capacity. Only new test wiring changed
  to the original guard. A retained-fixture run was **3 failed, 76 passed in
  11.31 s** because pytest's `current` symlink was selected; the original nofollow
  guard correctly refused it, and the test now selects actual candidate paths.
  Failed logs/results/source history and old tests are retained; final selection
  omits those three old capacity-blocked adapter positives without claiming pass.
  Read-only observations changed from 307200000 bytes at 07:32 UTC (original
  reserve-only guard refused) to 2293760000 bytes at 07:33 UTC; the original guard
  still refused the complete 9327844173-byte aggregate. No cleanup is attributed
  to this change; neither observation reserves space or certifies science storage.
  Latest local record `qec-ai-ci-evidence/matched-stage-parent-fullspace-observation.json`,
  SHA-256 `35ff383fbc8ffd5af95241e38fdf515ff14df178b017bc9b62dbc45980c6f858`.
  Tiny original codec/copy/receipt checks are real; study/host/UUID/lease/approval/
  clocks/capacity success and restored test caller Budgets remain synthetic.
  This is not a production outer runner, repository CI, full-private-suite pass,
  authenticated live caller or formal-readiness certification. No actual
  SSH/resource/GPU/container/preflight/transfer/study approval/sample/update/
  receipt operation was performed.
- Documentation PR #906 was merged by the user at 2026-10-26 07:13:15 UTC,
  merge `4bb6657d3004cb111439e1e711b9d7a90c24d3a3`. Exact head
  `5fbfa5394829f5467206e6e322e0437c51e4dd41` completed CPU run `40282109007`
  with 54 successful jobs and LocalGPU run `40282109149` with three successful
  and three skipped jobs. Local record `qec-ai-ci-evidence/pr906-user-merge-final-ci.json`
  retains job-level evidence. The agent did not merge or enable auto-merge; that
  retired branch is not reused. This CI does not certify the private acceptance API.
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
