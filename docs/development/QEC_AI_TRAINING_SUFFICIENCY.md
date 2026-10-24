# Prospective unchanged-model training-budget study

Protocol date: 2026-10-25. Publish this document before new optimizer updates or
new samples. This is a bounded training-response study, not a claim that a finite
budget proves convergence or improves logical error rate. The implementation and
new outcomes are pending when this protocol is first published.

## Question and unchanged controls

The [label diagnostics](QEC_AI_LABEL_IDENTIFIABILITY_RESULTS.md) found negative
raw logits and a descriptive gap between checkpoint BCE and within-sample feature
frequencies. Those observations do not identify the cause. Test whether extending
the existing optimizer trajectory changes held-out scores and correction activity
without changing the model, loss, features, optimizer or DEM. Retain every seed
and horizon, including negative or inconsistent responses.

Reuse the existing CPU `Trainer`, `Predecoder`, sampler, diagnostics and canonical
H/L. Restore the three audited A800-172 version-1 step-200 checkpoints from the
prior comparison. Preserve their model, Adam state, learning rate, batch generator
state and step counter. Do not reconstruct an equivalent-looking optimizer or
restart its random stream. Use the existing 81-parameter shared MLP and unweighted
mean binary cross entropy. Keep batch size 128 and the restored learning rate
0.001. No scheduler, weighting, resampling, architecture intervention or tuning.

Use exactly the original d9/R9/X/XV/p=0.003 independent training DEM, including
its disjoint-noise approximation and complete-signature parity merging. This
study does not validate that approximation against physical-circuit outcomes.
No stable API, checkpoint format, inference threshold or global decoder changes.
Reusable primitives remain in `decoders/_ai`; optional study orchestration remains
in examples. This is a CPU `single_device_fast_path`, without scalability claims.

## Fixed budget and order

| Stage | Frozen budget |
| --- | --- |
| Seeds | All model seeds 11, 22, 33; no seed selection or replacement |
| Input control | All three original step-200 checkpoints, file and tensor-state hashes, producing environment, DEM and optimizer state verified before new draws or updates |
| Optimizer trajectory | Continue each saved stream to absolute steps 600 and 2000; retain step 200, 600 and 2000 checkpoints; exactly 1800 new updates per seed, 5400 total |
| Training draws | Exactly 1800 x 128 new independent-DEM training shots per seed, 691200 total; no replay of the old training generator prefix |
| Held-out scoring | Exactly 4096 new independent-DEM shots per seed, generator seed = model seed + 12100000; the same fixed stream is replayed for each of that seed's three horizons |
| Scoring chunks | At most 32 shots; generators must be separate from the training generator |
| Calibration | Zero new fits or calibration samples; raw T=1 only |
| Validation selection | Zero validation samples and no horizon, temperature or threshold selection |
| Physical logical test | Zero new circuit-level or primary-LER shots; do not load the previous 65536-shot arrays |
| Fixed corrections | All-zero control and raw T=1 thresholds 0.5 and 0.9 at every horizon |

First complete exact software checks, then preflight all archived inputs and
versions, then execute all three fixed trajectories in seed order 11, 22, 33.
Score the step-200 control before its continuation. Save at step 600 and score,
then continue the same trajectory to step 2000 and score. Never stop early based
on held-out scores or choose which horizon to retain. The final horizon is fixed
at step 2000, not the lowest observed BCE. Held-out reports may be written during
the run but must not affect the declared trajectory. Record training loss for
every update, including nonmonotonic changes.

Repeated held-out streams across horizons are paired observations, not additional
independent samples. Mechanism labels within one shot are correlated and cannot
be counted as independent confidence observations. Seeds are distinct training
replicates, not a license to pool mechanism labels for narrow intervals.

## Environment, integrity and failure policy

Use the prior producing Python 3.12.13, Torch 2.13.0+cu130 and dependency lock in
image `sha256:9465e1fca0a8c22af1b2ee657c8a6ee7e9f9f8e74d873b0add84b274da58b9e2`.
Run CPU-only on an authorized A800 host in a fresh private directory with an
isolated auto-removed container, no GPU exposure or network, eight CPUs, 8 GiB
memory and one Torch thread. Inspect host availability immediately before launch.
Use a four-hour wall-clock limit for the complete formal study. Exhausting time
or memory is a retained incomplete result, not grounds for unregistered reruns.

Mount producing source and archived inputs read-only; create new outputs in a
fresh directory. Preserve the existing frozen comparison and diagnostic study
source, inputs, outputs and raw hashes. Do not overwrite old checkpoints. Log
source/protocol/image/dependency hashes, original checkpoint file/tensor hashes,
all stream seeds, exact step counters and output hashes. Missing, mismatched or
nonfinite inputs fail before new data. A nonfinite update or execution failure
aborts the study and retains partial records; no replacement seed, extra budget
or selective successful rerun. Do not treat a later evidence-only revision as
producing code.

## Required software acceptance before execution

Prove restoration preserves old logits, predictions and optimizer/batch state.
On a small fixed DEM, prove uninterrupted continuation equals a save/restore split
in model tensors, optimizer state, generator state and step count. Verify scoring
leaves the training generator and optimizer unchanged; chunked scoring reproduces
whole-stream sums and paired differences. Check all three input fingerprints
before any optimizer update, including refusal for a bad final-seed checkpoint.
Refuse reused output directories, wrong step counters, changed learning rates,
wrong DEM, incompatible versions and unregistered budgets. Fixture training is
software verification on a declared tiny synthetic DEM, never formal evidence.

Reuse the existing diagnosis runner's authoritative input validation and tensor
fingerprint where practical, rather than copying an independent validator. Do
not expand the core Trainer API merely to support study bookkeeping. The study
runner and its failure paths must be tested before the formal data collection.

## Report and bounded interpretation

For every seed and horizon report mean per-shot held-out BCE, positive/negative
BCE contributions, Brier score, positive-label prevalence, raw logit range,
positive-prediction counts, and all fixed-control H/L equivalence diagnostics.
Retain per-shot paired BCE differences against step 200, their mean, standard
deviation and quantiles, plus full exact-feature group counts and score summaries.
No asymptotic significance claim is registered here. Report the all-zero and
prior-only BCE references on the same shots as descriptive controls.

Declare a consistent descriptive training response only if step-2000 held-out
mean BCE is at least 20% below step 200 in **each** of the three seeds. Report
ratios for all horizons regardless of this outcome. Correction activation is a
separate descriptive endpoint: report whether any raw probability exceeds 0.5
and the fraction of shots with a nonzero proposed correction. It is not required
for the scoring-response designation and is never a logical-quality gate.

Passing this criterion supports only a response to this fixed additional budget.
Failure is negative or mixed evidence for this trajectory, not proof of an
architectural limit. Neither result proves convergence, training sufficiency,
recoverability, physical correctness, speed or LER improvement. Do not claim the
within-sample feature-frequency reference is a population Bayes optimum. Do not
change loss or expand the network inside this slice.

Any later logical-quality comparison requires another prospective protocol with
fresh independent training/calibration/validation/test streams, a shot budget and
statistical precision calculation, global-only and official Ising Fast on common
physical shots, and the existing paired quality gate. Official pretrained Fast
remains separate from equal-training-budget architecture research. No new Fast
inference or latency ranking belongs to this score-only study. The earlier
inconclusive official comparison and all negative results remain unchanged.

## Delivery acceptance

A complete research delivery requires the preregistered implementation, exact
resume/isolation/refusal tests, all fixed outcomes or an explicit retained
blocker, reproducible provenance, a verified user workflow and expected output,
and complete current-head CI. A protocol-only draft does not satisfy that gate.
No performance, quantization, export, domestic hardware or real-time stage is
unlocked by a lower loss. No PR in this slice is authorized for automatic merge.
