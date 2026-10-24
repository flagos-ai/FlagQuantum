# Fixed-budget calibration outcome

The [prospective protocol](QEC_AI_CALIBRATION.md) was published at
`7f52fa19b8ef6b0ef3bb352f288faae958713e27` before the full run. The experiment
executed the implementation at `b9a80ae0e3d4db3fd333f11ad2fe5a82ac3dbda1`
on Python 3.11.16 / PyTorch 2.10.0 CPU with one Torch thread. This report and
the checked-in [raw JSON](evidence/qec_ai_calibration_20261024.json) were added
after execution; no implementation or protocol was changed in response to the
outcome.

The study used all six trained models and all twelve raw/calibrated test
records, with 65536 shots per record and 196608 distinct shared test draws.
All training checkpoints and validation choices were frozen before testing.
No seed, control or threshold trial was removed and no samples were added.

## Observations

Every fitted temperature was 0.5. Validation BCE and Brier scores decreased in
all six models, but every validation threshold remained 1.0 and every test
logical prediction agreed with its baseline. Every paired record had zero
wins and losses; all quality gates remained **inconclusive**.

| Seed | Model | Positive labels | Positive share of training diagnostic BCE | Validation BCE, raw / calibrated | Validation Brier, raw / calibrated | Test failures, baseline / raw / calibrated |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | MLP | 2.263% | 51.09% | 0.07286 / 0.03685 | 0.008595 / 0.006457 | 10 / 10 / 10 |
| 11 | CNN | 2.263% | 64.80% | 0.06073 / 0.04317 | 0.007538 / 0.006502 | 10 / 10 / 10 |
| 22 | MLP | 2.250% | 29.87% | 0.12786 / 0.04122 | 0.016972 / 0.006883 | 10 / 10 / 10 |
| 22 | CNN | 2.250% | 58.55% | 0.07045 / 0.04412 | 0.008644 / 0.006937 | 10 / 10 / 10 |
| 33 | MLP | 2.350% | 42.60% | 0.09120 / 0.03993 | 0.010844 / 0.006667 | 8 / 8 / 8 |
| 33 | CNN | 2.350% | 62.55% | 0.06572 / 0.04350 | 0.008065 / 0.006693 | 8 / 8 / 8 |

All five validation candidates had equal logical failure counts within each
control, so the declared larger-threshold tie rule chose 1.0. The repeated
baseline failure counts across model families and controls refer to the same
physical samples; they are not extra independent observations.

Post-study inspection of the saved bounded-Tanh heads gives calibrated upper
logit bounds from 4.89 to 6.87, corresponding to probabilities below 0.999.
Thus the selected threshold 1.0 cannot activate these heads; it was not a
finite-precision sigmoid-saturation case. This inspection adds no sampled data
and does not change the frozen test decisions.

## Interpretation and limits

Labels are sparse, but positive labels contribute between 29.87% and 64.80%
of the diagnostic BCE. The observations do not support an explanation that
negative-label loss dominates every model. Reweighting needs its own controlled
study rather than following automatically from label frequency.

The held-out probability-score changes are descriptive. They do not establish
a logical-error-rate benefit, and temperature scaling preserves score order.
These fixed-grid controls also do not isolate calibration's causal effect on
LER. The rare-error budget leaves conservative ratio bounds imprecise. Preserve
the original study and this follow-up as inconclusive; the next research
decision concerns training/objective/ranking utility under a new declared
protocol. Performance optimization and hardware adaptation remain gated.

## Evidence and reproduction

- Raw JSON SHA-256:
  `70e72c4de5c4bc5029e50981071519632fbed307cf6a66490054336365f5e443`.
- Configuration SHA-256:
  `5a2ff62bc93a0025db6efe65ed9463f5cda47126fac9c450c63069d8a9034e40`.
- Protocol SHA-256:
  `4b85154a2fc2fa08471e79c82110b885d24f720666435f618cb36d77e5b5acc6`.

The JSON records every validation trial, loss partition, paired count,
configuration, source digest and six checkpoint digests. All recorded source,
protocol, configuration and checkpoint hashes were verified after execution.
The producing workspace retains the six training files and original manifest;
weights are not bundled into this source PR. Regenerate them with the fixed
command and a fresh output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python examples/qec/calibrate_predecoder.py \
  --output-directory /tmp/qec-calibration-full
```

This is a reproducible CPU research workflow, not a deployment package or a
guarantee of byte-identical Torch archive serialization across environments.
