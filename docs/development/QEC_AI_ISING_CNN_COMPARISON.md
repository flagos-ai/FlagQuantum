# Prospective Ising-layout CNN and matched-training comparison

Protocol date: 2026-10-25. Publish this protocol before implementing the new
study or collecting its training, scoring, calibration, validation or physical
samples. This document delivers a finite experimental design. The layout adapter,
matched-model runner, software acceptance and new outcomes remain unimplemented.
No new optimizer steps or experimental samples accompany publication.

## Question and limits

Does the existing native local-convolution design, adapted to the actual pinned
Ising detector layout, improve complete logical decoding under the same training
sample/update budget as a freshly initialized native MLP? Retain all three seeds.
The [continued-checkpoint outcome](QEC_AI_CONTINUED_PHYSICAL_ASSESSMENT_RESULTS.md)
was inconclusive for logical benefit; it does not establish that geometry, model
capacity, optimization or the training approximation caused that result.

Use exactly d=9, R=9, X memory, XV rotation and p=0.003 from the
[A800 protocol](QEC_AI_ISING_A800.md). Preserve its full canonical noise vector,
pinned circuit and DEM hashes, upstream source and official Fast weight hash.
Training retains the disclosed disjoint-noise independent-DEM approximation and
complete-signature parity merging. Physical validation/test use the original
categorical-noise circuit. Do not substitute a phenomenological memory model.

This is an equal-update/equal-sample comparison of two fixed native recipes, not
an equal-parameter, equal-FLOP or equal-wall-clock comparison. Official pretrained
Fast is a separate external reference with unverified training history. No
universal architecture ranking, causal attribution, timing benefit, convergence,
streaming capability or hardware promotion follows from this single cell.

## Existing capabilities and the concrete compatibility gap

`flagquantum/qec/decoders/_ai/spacetime.py` already implements a four-channel
syndrome/valid/X/Z encoder, two local 3D convolutions and incidence pooling onto
original DEM mechanisms. Its constructor currently takes a FlagQuantum
`MemoryCircuit` and phenomenological noise, constructs its own graphlike DEM and
assumes row-major square-grid data wires. It does not accept the actual Ising
circuit-level hyperedge DEM. The MLP already supports visible hyperedges through
explicit `allow_hyperedges=True`.

Reuse that encoder/pooling responsibility inside `_ai`; add only the private
construction/layout support needed for the verified Ising inputs. Keep vendor
inspection and conversion in optional examples. Preserve the existing constructor,
its default rejection behavior, old tensor names, public API and version-1
checkpoint payloads. Bind new layout/model identities in an external study
manifest; do not rewrite old checkpoint files or change their schema.
No second DEM, correction composer, sampler, trainer, statistics helper, artifact
manager or decoder registry is introduced by this protocol.

## Layout acceptance before any formal draws or updates

The pinned upstream builder in `qec/surface_code/memory_circuit.py` emits, for
this X-memory cell: 40 initial X detectors, eight subsequent 80-detector rounds
(each ordered Z then X, including the final perfect stabilizer round), then 40
terminal X detectors. There are 720 detectors, not ten complete 80-detector
slices. Its `add_detector` writes measurement-record parities without detector
coordinate annotations. Do not infer a grid by regular reshape or trust a
missing Stim coordinate table. The upstream evaluation transform is a reference,
not the native layout authority.

Construct a table in original flattened detector-index order. Resolve absolute
measurement records, measured ancillas, round and terminal data support from the
pinned circuit/schedule and its stabilizer matrices. Every detector must match
one unique X/Z stabilizer and one declared time. Independently check each parity
against the flattened circuit measurement records and canonical support ordering.
Reject unknown schedules, missing records, wrong boundary basis, rotations other
than XV, coordinate collisions or incomplete coverage before model execution.

Use the upstream check-ancilla coordinates, which are even integers in [0,18],
divided by two without rounding. The declared CNN volume is
`[time, row, column] = [10, 10, 10]`, with time 0 for initial checks, 1..8 for
subsequent stabilizer rounds and 9 for terminal checks. Record the axis convention
explicitly; verify it against qubit coordinates rather than assuming data-wire
index means x. Valid-site counts are 40, eight times 80, then 40. Preserve all
720 input bits, including the terminal complement detectors that the separate
mask-only diagnostic erases. No native input mask may silently apply that control.

Before formal execution require bounded zero/one/all-720-one-hot round trips,
exact valid/X/Z masks, independent detector-record/support/time checks and a
permuted-input/missing/duplicate-site refusal suite. Test X/Z and boundary
identity separately. A pointwise analytic encoder must pool onto each original
hyperedge column with exactly its complete incidence degree; independently XOR
selected original detector and logical signatures to check residual/frame bits.
Do not decompose hyperedges or average away their logical signature.

## Frozen models and training budget

Retain the existing CNN recipe: Conv3d(4,8,3,padding=1), ReLU,
Conv3d(8,8,3,padding=1), ReLU; mean incidence pooling; concatenate pooled eight
features with active count, degree and prior logit; Linear(11,16), Tanh,
Linear(16,1). There are 2817 trainable parameters. Retain the existing
Linear(3,16), Tanh, Linear(16,1) MLP with 81 parameters. Both use complete
identical H/L/prior buffers and score mechanisms from detectors alone. The CNN
uses past and future context and remains offline. No added layers, loss weights,
class balancing, feature ablations, schedulers or parameter search.

| Item | Fixed rule |
| --- | --- |
| Native models | CNN and fresh MLP, each at all seeds 11/22/33; no warm start from old checkpoints |
| Initialization | CPU FP32; reset Torch initialization seed to the model seed before constructing each model; retain initial tensors and RNG state |
| Optimizer | Existing CPU Trainer, Adam lr=0.001, unweighted mean mechanism BCE, batch=128, exactly 2000 updates per model |
| Training stream | Generator seed = model seed +18100000; 2000 calls of 128; replay identical draws for both architectures and verify every latent/input hash |
| Retained horizons | Steps 0, 200 and 2000 for every model; final physical contender fixed at step 2000 |
| Independent training draws | 256000 per seed, 768000 across three seeds; architecture replay is not additional independent data |
| Optimizer/sample exposures | 12000 updates and 1536000 shot exposures across six models; no append or replacement trajectory |
| Held-out scores | 4096 new independent-DEM shots per seed, seed+19100000, 128 calls of 32; same saved stream for both architectures/all three horizons |
| Calibration | Final step-2000 models only; 4096 new independent-DEM shots per seed, seed+20100000, 128 calls of 32, shared across architectures |
| Temperature grid | 0.25/0.5/1/2/4; reuse existing BCE fitter and exact tie rule |
| Validation | 131072 shared original-circuit shots, seed 21000011, one sampler, 32 calls of 4096 |
| Native threshold grid | 0.5/0.7/0.9/0.99/1.0 for each of twelve final native raw/calibrated paths; minimum full logical failures, ties choose larger threshold |
| Post-choice parity probe | One capture of 64 physical shots, seed 23000011, plus complete zero/one/one-hot detector rows; same saved inputs on both hosts |
| Primary | 4194304 shared original-circuit shots, seed 22000011, one sampler, 1024 calls of 4096; no optional stopping |
| Timing | Zero timing repetitions, throughput gates or optimization experiments |

Each training generator starts fresh; its state must not depend on model
initialization, scoring or the other architecture. Two Trainer instances may
replay the same declared stream, but their complete batch hashes must match at
every update. Count paired architecture/horizon reuse correctly. Do not pool
seeds or use mechanism labels as independent physical observations. Persist all
per-update losses, checkpoints, scoring/calibration labels and logits, and every
one of the 60 threshold trials. Scoring/calibration leaves model, optimizer and
training-generator states unchanged. An inactive threshold of 1.0 is retained.

Only training updates fit weights, only calibration fits temperature and only
validation selects thresholds. Freeze all choices and their external hash before
constructing the probe or primary sampler. Preserve Fast's pinned settings.
Old 65536-shot and 4194304-shot arrays and all old held-out arrays are excluded
from new inputs and cannot tune the recipe, threshold, horizon or budget.

## Primary family and decisions

There are fifteen prediction paths: twelve native paths (two architectures,
three seeds, raw T=1 and calibrated), global-only, official Fast and mask-only.
All paths see exactly the same physical shots and use the existing complete
local residual/frame plus shared global-matching composition.

Register exactly **32 comparisons**: twelve native paths each versus global and
Fast (24); CNN versus the matched MLP for every seed and raw/calibrated mode (6);
Fast versus global (1); mask-only versus global (1). Report every comparison,
including inactive and negative results. Step-200 checkpoints are descriptive
training controls, not alternate physical-test winners.

Use the already validated `exact_statistics.summarize_exact` with explicit
comparison alpha `0.05/32`. Each marginal gets alpha `0.05/64`, each marginal tail
`0.05/128`. Use the outward Clopper-Pearson ratio bound Ucandidate/Lreference;
zero reference lower bound is undefined/inconclusive. Keep paired counts/p-values
and unchanged legacy Hoeffding summaries. No new numerical interval method,
SciPy runtime dependency or post-result switch. Test this family allocation and
all 32 identities before formal data; numerical certification failure stops the
study without fallback or an apparent successful summary.

The primary CNN quality gate requires all three calibrated step-2000 CNN seeds
to have ratio upper <=0.9 versus global. A separate incremental matched-recipe
benefit requires that conjunction and all three corresponding CNN/MLP upper
bounds <=0.9. A separate competitive-with-Fast statement requires all three
CNN/Fast upper bounds <=1.05. All conjunctions and all MLP controls are reported
regardless of outcome. Failure to certify is inconclusive, not inferiority.

For precision planning only, choose a hypothetical baseline rate 0.00125,
without loading old test arrays. At N=4194304 this means 5242.88 expected failures.
The registered tail allocation gives normal z approximately 3.359354. Normal
plug-in ratio uppers at candidate/reference true ratios 1/0.9/0.8/0.7 are about
1.097241/0.989886/0.882389/0.774720. These are planning approximations, not exact
intervals, measured effects, formal power or guaranteed feasibility. A ratio at
the 0.9 target cannot generally be certified with high probability. The fixed
budget remains final even if realized rates make the answer inconclusive.

## Execution and failure gates

Implement and review the layout/model slice first, then the narrow matched-study
composition and refusal tests using existing Trainer, archive validators,
calibration, paired statistics and continued-assessment shard/parity machinery.
Prove tiny synthetic training/resume equality, all-six-input validation before
updates, matched batch hashes, scalar prediction/count oracles, choices-before-
primary ordering and nonfinite/partial/duplicate/timeout refusal. Synthetic
fixtures are software verification, never evidence of trained scientific benefit.
Do not launch the formal study until complete workflow tests, final producing-head
CI and matching-image zero-sample preflight pass. Freeze producing source,
protocol, layout, canonical DEM, environment and new initial-state hashes first.

Use the previously pinned Python 3.12.13 / Torch 2.13.0+cu130 image and dependency
lock from the A800 protocol. Native training/calibration are CPU-only; physical
inference is FP32 with deterministic settings, no TF32/autocast and batches <=32.
Both hosts must pass exact CPU/CUDA residual/frame and physical-final-prediction
parity on the same frozen probe, on distinct actual GPU UUIDs, before one primary
capture on a800-172. CPU correctness fixtures cannot replace actual receipts.

A single formal launch has a 24-hour total hard deadline covering training
through final reduction. Recheck actual idle CPU/GPU resources before launch;
use fresh private directories, read-only source/whitelisted inputs, isolated
network-none auto-removed containers, eight CPUs, 16 GiB and one Torch thread.
Use the existing exclusive controller/launch locks, authenticated exact artifact
handoff and owned-container-only timeout cleanup. Do not mount old study roots
or old sample arrays; authorized vendor code/dependencies/weights may be mounted
individually read-only. Retain all raw shards/predictions, source/input hashes,
choices, receipts, logs and complete failure records. Independently audit stored
predictions and all 32 tables after completion with zero new samples/updates.

Time, memory, nonfinite or integrity failure retains an incomplete study and
partial files; no restart, successful-seed replacement, model reduction, extra
shots, loss change or amended significance threshold. An insufficient fixed
budget is an outcome. Preserve every earlier study/source/checkpoint unchanged.
No performance, export, quantization, domestic-accelerator or real-time stage is
advanced unless a separately registered scientific quality gate is established.

## Executable budget check

This standard-library example only checks declared arithmetic. It does not run
an unimplemented runner, construct models or draw data.

```python
from statistics import NormalDist

seeds, models, steps, batch = 3, 2, 2000, 128
native = models * seeds * 2
comparisons = native * 2 + seeds * 2 + 2
print(seeds * steps * batch, models * seeds * steps * batch)
print(native + 3, comparisons)
print(f"{NormalDist().inv_cdf(1 - 0.05 / comparisons / 4):.6f}")
```

Expected and observed output at protocol publication:

```text
768000 1536000
15 32
3.359354
```
