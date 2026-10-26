# Correction-label identifiability diagnostic outcome

The fixed archived-checkpoint diagnostic completed on all three seeds. Every
candidate control was inactive, including raw probability thresholds 0.5 and 0.9.
The exact feature table contains mixed labels and some high observed positive
fractions, while the saved scorer gives only negative logits on this batch.
This motivates a separately frozen training-sufficiency study; it does not prove
its cause, architecture inadequacy, a population error floor or a logical benefit.

## Producing code and fixed budget

The [prospective protocol](QEC_AI_LABEL_IDENTIFIABILITY.md) was published before
new diagnostic sampling. Producing source is
`7384185fc297ed7410637c6f4a45a06f82ffeb35`; the later evidence/report commit must
not be substituted for that source. All recorded source-file and protocol hashes
were rechecked after collection, without generating new outcomes.

Reuse the A800-172 d9/R9/X/XV/p=0.003 independent training DEM and its existing
step-200 checkpoints (model seeds 11/22/33). Each seed received exactly 4096 new
DEM samples at sampler seed +9100000, processed in chunks of 32. Controls were
all-zero, raw T=1 at thresholds 0.5 and 0.9, and the previous frozen calibrated
choice. There were zero new optimizer steps, calibration fits or primary logical
test shots. The original circuit-noise independent approximation remains a
limitation; these are training-distribution diagnostics, not physical-circuit
validation. No Fast inference or new performance measurement was performed.

Each stream contains 48738304 mechanism labels. Those labels within a shot are
correlated and are not 48738304 independent calibration observations. All three
seed streams and every control are retained; no group or seed was selected to
change the protocol or obtain significance.

## Preflight and environment

All metadata files were checked against the retained source-study archive.
Checkpoint file and tensor-state hashes, canonical DEM, version 1, step 200,
producing Python/PyTorch and dependency versions were verified for **all three
seeds before any new real-workload sample**. Exact-oracle and runner tests then
passed in that same isolated environment: 27 passed. The tensor fingerprint was
also checked against independent native-byte construction in the tests.

The single CPU-only, network-disabled, auto-removed container used image
`sha256:9465e1fca0a8c22af1b2ee657c8a6ee7e9f9f8e74d873b0add84b274da58b9e2`,
Python 3.12.13, PyTorch 2.13.0+cu130, one Torch thread, an eight-CPU limit and an
8 GiB memory limit on A800-172. No GPU was exposed. Source, archived checkpoint
copies and dependencies were mounted read-only. The input subset contained only
required metadata/DEM/checkpoints; old primary-test arrays were not mounted.
The unique bounded job exited 0; it was not retried or extended.

Raw manifests and per-seed records are archived byte-for-byte under
[evidence](evidence/qec_ai_label_diagnostics_20261025/manifest.json).
The [post-run audit](evidence/qec_ai_label_diagnostics_20261025/audit.json)
records all four output hashes, source/bundle/image identifiers and integrity
checks. Full launch/preflight logs and the immutable source bundle remain in
`/tmp/fq-ai-label-study-20261025-7384185` on the producing host; local raw copies
are under `/Users/baai/Documents/liuwei/qec-ai-label-diagnostics-20261025`.

## Loss and score diagnostics

| Seed | Positive labels | Positive fraction | Raw mean BCE | Frozen-calibrated mean BCE | Empirical group-frequency BCE | Negative share of raw BCE | Maximum raw logit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 60594 | 0.00124325 | 0.04676592 | 0.00987348 | 0.00261092 | 91.39% | -2.847650 |
| 22 | 60341 | 0.00123806 | 0.08503267 | 0.01256319 | 0.00257079 | 96.21% | -2.027288 |
| 33 | 60562 | 0.00124260 | 0.06785757 | 0.01123143 | 0.00259501 | 94.87% | -2.482098 |

Raw predicted-positive counts were zero for every seed. Every maximum raw logit
is negative, so positive temperature scaling cannot move these scores above the
0.5 probability threshold. The smaller calibrated aggregate BCE does not create
active corrections. The negative-label loss shares apply to this d9 diagnostic
distribution; earlier studies and their different distributions remain unchanged.

The empirical group-frequency BCE is an in-sample descriptive reference obtained
from exact feature groups. It is not a held-out model, an estimator of a
population irreducible loss, a proof of optimizer convergence or an accuracy
bound. Its gap from the saved checkpoint scores warrants a controlled learning
study before assuming that feature expansion is necessary.

## Exact feature groups

| Seed | Groups | Groups with both labels | Observation share in mixed groups | Highest observed positive fraction |
| --- | ---: | ---: | ---: | ---: |
| 11 | 263 | 218 | 89.87% | 0.9758 |
| 22 | 263 | 216 | 62.35% | 0.9935 |
| 33 | 263 | 219 | 62.87% | 0.9714 |

There were no singleton groups. Every group's counts, positive fraction, raw
score/probability means and logit range are retained, including pure-negative
and rare groups. The highest observed fraction is descriptive and was not used
to choose a threshold, feature or intervention. Group observations do not supply
independent confidence trials; report no population posterior claim from these
fractions. Equal features can carry different latent labels even when learning
can still exploit their conditional distribution.

## Correction equivalence

For every seed and all four controls, latent-vector equality, detector equality,
joint detector/logical equality and detector-equal/logical-different counts were
all **zero out of 4096**. Hamming mismatch bits were 60594/60341/60562 for seeds
11/22/33, respectively: every control returned the all-zero correction.

The exact small-cycle oracle demonstrates that different latent vectors can
have equal H/L output, and separately rejects a cycle with a logical flip. The
real-workload candidate comparison did not observe such a latent-mismatched but
H/L-equivalent correction in this fixed sample. It therefore does not establish
that label degeneracy caused the native model's inactivity. These comparisons
are against complete latent truth: an offline predecoder can leave a syndrome
for its global decoder. Zero H/L equality here is **not** a 100% final logical
decoding failure rate.

## Validation and next gate

Focused local QEC tests: 133 passed, one real-CUDA test skipped locally. The
27 exact/runner tests also passed inside the producing CPU environment. Strict
mypy, Ruff, Black, architecture, English-language policy, example inventory,
team ownership and whitespace checks passed. Local `pr-default` exceeded its
120-second bound and is not a pass. Local revision-history certification is
unavailable in the shallow checkout; complete-history, actual-head CI remains
required. Evidence field-name and inline-origin checks passed locally.

The next scientific slice should predeclare a fixed training-sufficiency budget
for the unchanged model, retain the step-200 controls and independent held-out
streams, and measure score/operating-point changes before choosing a loss or
feature intervention. Any new logical comparison needs its own prospective
sample budget and precision analysis, including the official Fast and global
baselines. Do not reopen the old 65536-shot test or append samples until a gate
passes. No accuracy, speed, export or domestic-accelerator gate is promoted by
this diagnostic.
