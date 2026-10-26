# Prospective predecoder diagnostics and calibration

Protocol date: 2026-10-24. This follow-up starts from the merged paired study
(PR #786) and the private-package consolidation (PR #795). This document is
committed before collecting follow-up test results. It does not amend the first
study or promote a capability.

## Motivation and scope

The first study reported 54 groups, 16384 shots per group, thresholds of 1.0,
equal hybrid/baseline logical failures and inconclusive gates. Its raw JSON has
SHA-256 `c43020b8ec44cc9fdceec218a69208274a5f579c0690aa74bf30a30d3ab01d2e`;
configuration SHA-256 is
`5c9fedb301ad2b49dd956bf411dfacd735e7287543dd753fbc8665da7b20ea71`.
Those results remain unchanged. Original model checkpoints were not archived;
retraining below is a declared new experiment, not replay of archived weights.

This round diagnoses label sparsity and positive/negative BCE contributions,
fits one scalar temperature per model using calibration data, and selects a
confidence threshold using separate validation data. It reuses the existing
DEM, sampling, MLP/CNN, Adam trainer, residual/global decoder and paired statistics.
It does not introduce a different training loss, accelerator, streaming path or
performance claim. Diagnostics will inform a separately declared loss study if
needed. Implementation remains in `decoders/_ai/`; public exports and training
checkpoint contents remain unchanged.

## Fixed experiment and budget

All runs use CPU, Z-memory, independent phenomenological data-X and measurement
flips and the existing PyMatching adapter. Preserve every seed/model/control.

| Stage | Fixed setting |
| --- | --- |
| Model seeds | 11, 22, 33; shared local MLP and geometric 3D CNN |
| Training | d=3, rounds=3, p=0.02; 200 Adam steps, batch 128, learning rate 0.001; sampler seed = model seed +104729 |
| Loss diagnostics | Training distribution; 4096 fresh shots; seed = model seed +3000000 |
| Calibration fit | d=3, rounds=3, p=0.006; 4096 shots; seed = model seed +4000000 |
| Threshold validation | Same d/rounds/p as calibration; 4096 separate shots; seed = model seed +5000000 |
| Primary test | d=5, rounds=5, p=0.003; 65536 shots; seed = model seed +6000000 |

There are six trained models and two inference controls per model: temperature
1.0 and the calibration-selected temperature. Each control selects its own
validation threshold before any primary test is evaluated. This gives twelve
paired test records at 65536 shots each. Identical test streams are reused across
both controls and both model families within each seed: there are 196608 distinct
test draws, not 786432 independent test observations. Do not pool seed/model
records into an independent binomial experiment.

Training work is bounded at six times 200 times 128 model-shots. Diagnostics,
calibration and validation each use six times 4096 model-shots. There is no
optional stopping, best-seed selection, adaptive test-size increase or extra
test cell. Preserve partial/failed runs and use a new output directory rather
than overwriting them. Smoke runs use seed 11, two steps, batch 16 and 32 shots
for each held-out stage; they test software only and never replace full results.

## Temperature and threshold rules

Temperature fitting minimizes mean mechanism BCE on calibration data over exactly
T in {0.25, 0.5, 1.0, 2.0, 4.0}. Ties prefer the temperature closest to 1.0 in
absolute log distance, then the smaller temperature. No label from threshold
validation or testing fits the temperature. Temperature applies as logits/T at
inference; weights and DEM buffers do not change.

For each control, test validation thresholds {0.5, 0.7, 0.9, 0.99, 1.0} on the
same validation stream. Choose the fewest hybrid logical failures; ties choose
the larger threshold. Record every candidate's paired counts, including the
unselected ones. Do not assume threshold 1.0 disables correction: inspect the
actual counts and predictions, including finite-precision sigmoid saturation.

Transfer only the learned network/encoder to each target DEM/layout using the
existing study transfer routine. Keep its own probabilities, H/L and geometry.
Save all six training checkpoints before fitting or testing. The report binds
checkpoint files, the frozen protocol, configuration and implementation files
with SHA-256 digests. Temperature and threshold are explicit inference settings
in the research report, not a claim of an independently deployable model package.

## Interpretation

Diagnostics report positive/negative label counts and summed BCE contributions,
predicted-positive counts, mean confidence and Brier score. Record both raw and
scaled validation diagnostics on the same held-out batch. These are descriptive:
mechanism labels within a shot are not independent calibration observations,
and a small mean loss/Brier score can be dominated by negative labels.

The confirmatory endpoint remains the calibrated hybrid/global-baseline logical
error ratio at the primary cell. Use the existing conservative paired summary
with Bonferroni alpha=0.05/6 across two model families and three seeds. A family
must pass the 10% reduction bound at all three seeds to support that claim.
Raw-temperature controls and probability diagnostics are descriptive. Comparing
the two fixed-grid inference pipelines does not isolate temperature's causal
effect on LER: rescaling logits also changes the grid's operating points.

This bounded diagnostic budget is not a power guarantee for a 10% improvement
at rare logical error rates. Precision exhaustion, equal counts, no discordant
pairs or no useful selected corrections remain inconclusive. No loss decrease,
improved calibration score or smoke output satisfies the scientific benefit
gate. Retain all original and follow-up negative findings; optimization and
hardware work remain gated on scientific evidence.
