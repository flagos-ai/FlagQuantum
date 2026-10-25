# Prospective correction-label identifiability diagnostics

Protocol date: 2026-10-25. This protocol is published before collecting its new
samples or diagnostic outcomes. It follows the inconclusive
[d9 A800 checkpoint comparison](QEC_AI_ISING_A800_RESULTS.md); it does not amend
that study, increase its test budget or promote a model. Implementation is pending
in the draft follow-up when this protocol is first published.

## Question and scope

The shared MLP consumes the count of active incident detectors, mechanism degree
and logit prior. It does not consume detector identity or neighboring patterns.
The trainer supervises a sampled latent mechanism vector. Existing aggregate BCE,
label counts and calibration diagnostics do not establish that this latent vector
is uniquely recoverable from those features, or from the whole syndrome.

This slice separates three questions:

1. Are two different latent vectors equivalent under the complete detector **and**
   logical projections? A latent-label mismatch need not be a wrong correction.
2. Does the current exact feature representation group conflicting binary labels?
   Describe the observed conditional label distribution without claiming a
   population Bayes bound from a finite sampled table.
3. How do saved checkpoint scores compare with these diagnostic distributions?
   Descriptive diagnostics are not evidence of training sufficiency or a new LER
   benefit. Optimizer convergence and a larger training budget remain separate
   prospective experiments.

Keep all reusable diagnostics in `decoders/_ai/diagnostics.py`. Reuse the
canonical DEM H/L matrices, sampler, predecoder, existing loss diagnostics and
saved trainer checkpoints. If feature construction needs reuse, extract its
existing arithmetic without changing its result or adding public API. Do not
create another DEM, decoder registry or geometric layout. Do not add loss
weighting, train a new network, extend the native CNN layout, change the
calibration rule or optimize inference in this slice. Public APIs and checkpoint
contents remain unchanged.

## Exact correctness fixtures

Before real-workload diagnostics, exhaustively enumerate fixed small independent
DEMs with at most 12 mechanisms (at most 4096 latent states each). Cover:

- an injective complete H/L mapping;
- a detector cycle whose nonzero kernel vector also has zero logical projection;
- the same detector cycle with a nonzero logical projection, so syndrome equality
  alone is explicitly insufficient;
- mechanisms sharing degree/prior and active-detector count while differing in
  identity or neighborhood, with both positive and negative labels;
- complete hyperedges and an empty logical-observable dimension.

Use analytic XOR projections independent of the diagnostic implementation as the
oracle. Preserve original complete signatures; do not decompose hyperedges.
Where probabilities are reported for these fixtures, enumerate their independent
Bernoulli probabilities and verify normalization. Clearly distinguish exact
fixture probabilities from empirical real-workload frequencies.

Check finite binary values, compatible shapes, CPU/device refusal boundaries,
exact zero-length handling and removal of any temporary instrumentation. All
candidate/target projections must agree with the canonical H/L composition.
A model/checkpoint reload must preserve logits and correction bits. No test may
turn a logical-frame mismatch into equivalence merely because syndromes match.

## Fixed real-workload budget

Use the existing d=9/R=9/X/XV/p=0.003 independent training DEM from the frozen
A800 study, retaining its full noise vector, disjoint-alternative approximation
and complete-signature parity merging. This is training-distribution diagnosis;
it is not an independent physical-circuit test or a check of the approximation.

| Stage | Fixed budget and ordering |
| --- | --- |
| Inputs | Use the three archived native checkpoints from A800-172, seeds 11/22/33, each at step 200; verify their prior audit/tensor-state hashes before use |
| New diagnostic samples | Exactly 4096 independent-DEM shots per checkpoint, generator seed = model seed +9100000; process in chunks of at most 32 |
| Candidate latent corrections | All-zero control; raw T=1 with thresholds 0.5 and 0.9; previously frozen calibrated temperature/threshold from the A800 choices file |
| Exact features | Use the same floating-point feature triples presented to the MLP; no quantization, approximate bins, rounding or label-dependent grouping |
| Training | Zero new optimizer steps; no checkpoint replacement or best-seed selection |
| Calibration and validation | Zero new fits or threshold selection |
| Primary logical test | Zero new shots; do not re-open or tune against the previous 65536-shot test |
| Hardware | CPU-only diagnostics in an isolated environment matching the producing checkpoint environment; no GPU timing or performance ranking |

Finish all exact oracle tests before new real-workload collection. Missing or
mismatched archived checkpoints/DEM/choices fails closed rather than retraining.
Read the frozen source directory without modifying it; run follow-up code from a
separate task directory. Archive the new protocol/source/input hashes and
versions, stream seeds, all three seeds and every control. A failure aborts the
run and retains partial evidence; retries that would create new outcomes require
a new declared protocol rather than selecting a favorable completion.

A smoke mode uses only the exact fixtures, or at most 32 new diagnostic DEM
shots at seed 11 +9100000 with the same controls. It must be labeled software
verification and cannot replace the full fixed diagnostic budget. It never
requires authorized weights to be copied into Git. No new Fast inference is
needed for this latent-label question: retain the previous official-checkpoint
baseline and its inconclusive quality results rather than inventing a new
architecture comparison without a common endpoint.

## Required report and acceptance

For each seed and candidate, report exact latent-vector equality, complete
H-projection equality, joint H/L-projection equality and detector-equal but
logical-different counts. Report latent Hamming mismatch separately. State that
these are mechanism-supervision diagnostics: a predecoder can leave a residual
for the global decoder, so a mismatch against the latent truth is not by itself
a final logical decoding failure.

For each exact feature group, retain total observations and positive/negative
counts, empirical positive fraction and the candidate score summary. Report
singleton groups, groups containing both labels, their observation share and
empirical conditional binary entropy/BCE at the empirical group frequency.
The empirical optimum is a within-sample descriptive reference, not held-out
calibration, an estimator of population irreducible loss, or proof that feature
ambiguity caused the observed LER. Correlated mechanisms within a shot must not
be counted as independent confidence observations. Preserve the existing summed
positive/negative BCE accounting beside group-level metrics.

Accept this engineering slice only when exact projection/feature oracles pass,
old CPU/checkpoint behavior is unchanged, all fixed controls and samples are
retained or a reproducible blocker is documented, a runnable user example and
expected output are verified, and current-head CI is complete. Keep the two
independent A800 runs and the quarantined host-RSS fields of the previous study
unchanged. Do not infer an accuracy, speed, real-time or hardware release benefit
from this report.

The outcome may justify a separately preregistered training-sufficiency study or
one precisely scoped feature/model intervention. Neither a failed checkpoint
nor a finite feature table licenses adaptive extra training, test sampling,
loss changes or jumping ahead to optimization/export/domestic acceleration.
