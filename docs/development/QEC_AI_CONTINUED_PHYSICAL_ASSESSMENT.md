# Prospective physical assessment of continued checkpoints

Protocol date: 2026-10-25. Publish before any new calibration, validation,
device probes or primary test collection. Implementation and outcomes are pending
when this protocol is first published. The [training-response outcome](QEC_AI_TRAINING_SUFFICIENCY_RESULTS.md)
showed a descriptive score response; it did not establish logical benefit.

## Question and fixed contenders

Do the already frozen step-2000 checkpoints improve complete physical-circuit
logical decoding after independent calibration and validation-only threshold
selection? Retain step-200 controls and all model seeds 11/22/33. Do not select a
horizon or seed from the prior BCE/activation results. No new optimizer steps,
loss changes, features, CNN, precision intervention or inference optimization.

Reuse the exact d9/R9/X/XV/p=0.003 workload, canonical full noise vector, pinned
upstream source and authorized Fast weight hash in
[the original A800 protocol](QEC_AI_ISING_A800.md). Verify circuit and training
DEM hashes against that archive before any new data. Training/calibration labels
retain the disclosed independent approximation of disjoint noise alternatives
and complete-signature independent-parity merging. Validation and test use the
original categorical-noise circuit sampler, not independent-DEM samples.

The fourteen fixed prediction paths are:

- global-only uncorrelated batched CPU PyMatching;
- actual official pretrained Fast with its frozen upstream threshold settings;
- native raw T=1 and independently temperature-calibrated paths, for each of
  three seeds and each of steps 200 and 2000: twelve paths in total.

Retain the terminal-complement mask-only path as an additional diagnostic
control. Use the same global-decoder implementation and settings for every
residual. Preserve complete H/L and upstream input/frame semantics. Raw and
calibrated threshold choices are independently frozen from validation only.
Official Fast is an external pretrained checkpoint, not an equal-training-budget
architecture control; do not describe this as architectural superiority.

## Inputs and preflight

Load the six existing step-200/2000 checkpoints from the single completed
training-response study. Verify all file and tensor-state fingerprints using its
public audit and prior full-content checkpoint audit. The original step-200
contents must match their old A800 originals. All six inputs, checkpoint version,
DEM, model type, exact horizon and producing dependency versions must pass before
new samples. Do not retrain missing checkpoints or substitute a seed/horizon.
Do not mount any old primary physical-test arrays or old held-out score arrays.

Reusable AI statistics/calibration remain in `flagquantum/qec/decoders/_ai`;
optional vendor/device/study composition stays in examples. Reuse existing
`PairedCounts`, `paired_counts`, `paired_p_value`, temperature diagnostics,
canonical DEM, sampler, layout oracle and Ising adapter. Preserve public APIs,
old checkpoint schema/content, and previous assessment defaults/results.

## Frozen streams, budgets and selection order

| Stage | Fixed budget and stream |
| --- | --- |
| Training | Zero new updates; all six checkpoint identities fixed before data |
| Calibration | 4096 independent-DEM shots per seed, seed = model seed +14100000; replay the same stream for both horizons; inference chunks <=32, no gradients |
| Temperature grid | 0.25/0.5/1/2/4, minimum mean BCE; ties use minimum absolute log temperature, then smaller temperature, exactly as the existing fitter |
| Validation | 131072 shared original-circuit shots, seed 15000011; replay exactly the same fixed chunk sequence for all native trials |
| Threshold grid | 0.5/0.7/0.9/0.99/1.0 for each of twelve native paths; fewest complete logical failures, ties choose larger threshold, including inactive 1.0 |
| Device probes | All detector one-hot inputs plus zero/one rows; 64 physical circuit shots at seed 17000011, after every choice freezes and before primary sampling |
| Primary physical test | Exactly 4194304 shared original-circuit shots (2**22), seed 16000011, fixed sampler calls of 4096 shots; no optional stopping |
| Neural inference | FP32, batch <=32, same physical A800 device on a host for every contender |
| Timing | Zero new timing repetitions or performance gates in this quality slice |

Only calibration labels choose temperature. Only the independent physical
validation stream chooses thresholds. Do not change Fast thresholds. Persist
all calibration scores, every validation trial, selected choices and their hashes
before constructing the primary sampler. A chosen threshold of 1.0 is valid and
must not trigger a lower threshold, extra training or replacement model.

For each stage construct one sampler with its declared seed and fixed call
sizes, retain its chunk order and input hashes, and replay saved chunks rather
than silently changing call segmentation between contenders. Each calibration
stream is 128 calls of 32 shots; validation is 32 calls of 4096; primary is 1024
calls of 4096. Distinct stream seeds do not imply disjoint physical events.
All twelve native paths, Fast, global-only and mask-only see the exact same
primary physical samples. Repeated checkpoints, seeds, modes or hosts do not
multiply 4194304 into independent samples. Store every prediction and truth bit
in bounded immutable shards with a complete manifest, not only final counts.

The fixed zero-training exact-fixture smoke uses synthetic DEMs and analytic
XOR/counts only. No exploratory d9 data or alternative physical test budget is
registered. Preflight bit probes are correctness evidence and cannot substitute
for the primary quality experiment or tune choices.

## Statistical method declared before data

The previous assessment uses distribution-free Hoeffding marginal intervals.
At rare logical-error rates these can be too wide to establish the same 10%
ratio target with a practical budget. Keep those summaries unchanged and report
them as a reference. Do not reanalyze old frozen experiments to promote a gate.

For this new protocol add a separately named private exact-binomial interval
and summary path before any new data. Use equal-tailed Clopper-Pearson marginal
intervals, obtained by inverting binomial tails, including exact zero/all-failure
endpoints. [SciPy's exact method](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
is an independent numerical reference, not a required new runtime dependency.
Do not use Wilson/normal intervals, mid-P or bootstrap for the formal gate.

Register a family of **26 comparisons**: each of twelve native paths versus
global-only and versus Fast (24), Fast versus global-only (1), and mask-only
versus global-only (1). Use family alpha 0.05 and comparison alpha
`a = 0.05 / 26`. Within a comparison, give each of the two marginal intervals
miscoverage `a/2`, hence each marginal tail is `a/4`. If candidate interval is
`[L_c,U_c]` and reference interval `[L_r,U_r]`, use the conservative ratio bound
`U_c/L_r`; return undefined/inconclusive when `L_r == 0`. By a union bound this
does not require independent candidate/reference errors or independent
comparisons. Preserve paired wins/losses and the existing exact two-sided
McNemar discordant-pair p-value; a p-value alone never satisfies the benefit gate.

The primary native logical-quality decision requires **all three step-2000
calibrated seeds** to have the registered ratio upper bound <=0.9 versus
global-only. Other paths are fixed controls, not fallback winners. Report all
26 comparisons and all step-200/2000 differences even when this conjunction
fails. A separate "competitive with official Fast" statement is permitted only
if all three primary native paths have ratio upper bound <=1.05 versus Fast.
That statement is not an equal-budget architecture claim. Failure or insufficient
precision is inconclusive; do not lower a target or increase alpha afterward.
No speed gate, optimization stage or production promotion is tested here.

Validate the exact interval before new calibration or physical collection:
analytic zero/all-failure endpoints and reflection; independently enumerated
small-n binomial-tail inversion and coverage across a fixed p grid; monotonicity,
finite rare-event large-n cases and numerical agreement with a pinned independent
reference. Use outward conservative numerical bracketing, reject nonfinite or
unvalidated results, and test zero reference failures explicitly. Legacy
`rate_interval`/`summarize` defaults and their regression expectations remain
unchanged. If the exact primitive cannot be certified, stop before new data;
do not choose an interval method after seeing test outcomes.

## Precision and feasibility analysis

The previous aggregate 83/65536 failure rate is used only as a planning anchor,
not to tune a model or select a new test outcome. At that rate, 2**22 new shots
would yield about 5312 baseline failures in expectation. A normal approximation
with the registered marginal tail allocation gives z approximately 3.301545.
For planning only, expected conservative ratio bounds at candidate/reference
true ratios 1.0/0.9/0.8/0.7 are approximately
1.09483/0.98766/0.88035/0.77287. These are approximations, not exact intervals,
power guarantees or predictions of native benefit. The exact gate uses realized
counts and the prevalidated exact method above. A true ratio at the 0.9 boundary
cannot generally be certified as <=0.9 with high probability from finite data.

At the same budget, the old Hoeffding marginal radius is approximately 0.000954,
which is large relative to the planning baseline rate 0.001266. This motivates
the declared, separately tested exact method rather than a post-result switch.
If actual failure rates are lower or discordance differs, even this larger fixed
budget may remain inconclusive. Do not run a pilot to revise the primary budget,
append chunks, pool hosts or stop early at significance. Four million shots are
a fixed scope, not a commitment that a scientific gate will pass.

## Device, resources and failure policy

Use the same pinned image/dependency environment as the producing training
study. Calibration remains CPU with no gradients. Run native and Fast inference
on one actually idle A800 on A800-172, FP32, TF32/autocast disabled and deterministic
algorithms enabled; retain CPU global matching. Record image, source/protocol,
checkpoint/weight hashes, device UUID/model/driver/runtime and CPU/thread limits.
Use private fresh directories and isolated auto-removed containers, no network,
16 GiB host-memory limit, eight CPUs and one Torch thread. Bound the complete
formal calibration/validation/primary workflow to 24 hours, retaining partial
outputs and stopping only this task's workers on failure or timeout.

After every choice freezes, require CPU/CUDA residual and logical-frame equality
on all one-hot/zero/one probes, then final logical-prediction equality on the 64
physical shots. Arbitrary synthetic syndrome rows need not admit matching.
Repeat the same frozen bit probes on an available A800-171 before primary
sampling; those repeated probes are not additional quality evidence. A mismatch
or unavailable required input/device blocks primary collection; do not tune,
silently fall back, regenerate probes or omit a seed to make it pass. There is
one primary host and no second-host primary experiment or timing ranking.

Freeze producing source, all six input checkpoint hashes, image/dependency lock,
calibration/validation choices, exact statistical implementation and protocol
before primary data. Mount source and inputs read-only. Any failure aborts and
retains partial shards; no retry producing a different formal outcome. Existing
completed study directories, binary arrays and raw reports remain untouched.
Optional fixture tests are software evidence only. No paid service or real QPU.

## Required delivery and acceptance

Before execution, exact tests must prove interval coverage/refusal boundaries,
checkpoint/DEM/version/hash preflight for every seed, selection/tie rules,
independent random streams, shared physical chunk order, bounded memory/shards,
immutable frozen choices before primary sampler construction, CPU/CUDA bit
semantics, and all control/prediction counts. Exercise failure before primary
sampling on corrupted late-seed inputs, wrong weight hash or parity mismatch.

Deliver all calibration/validation trials, selected choices, bit-probe evidence,
complete primary shards/predictions, paired counts/registered summaries, negative
and inconclusive outcomes, producing revision/hash provenance, a verified user
example and expected output, and final-head full CI. Public APIs/checkpoint
schemas and the single DEM/decoder authority stay unchanged. A protocol-only
draft is not implemented scientific delivery. No automatic merge is authorized.
