# Fixed-budget unchanged-model training response

Outcome date: 2026-10-25. The [prospective protocol](QEC_AI_TRAINING_SUFFICIENCY.md)
was published before any new research updates or held-out samples. This outcome
retains every seed and horizon from the single completed run. It does not amend
any previous logical-quality comparison or advance a deployment gate.

## Result and decision

All three seeds passed the preregistered **descriptive training-response** rule:
step-2000 held-out mean BCE was at least 20% below its step-200 control. Observed
reductions were 90.61%, 93.02% and 92.51% for seeds 11, 22 and 33. These are paired
score descriptions on the fixed independent-DEM streams, not statistical proof
of convergence, training sufficiency or logical-error reduction.

At raw T=1 and threshold 0.5, seeds 11 and 33 proposed nonzero corrections on
2524 and 3095 of 4096 shots, respectively; seed 22 remained inactive. Threshold
0.9 remained inactive for all seeds and horizons. No calibration, validation
selection, physical logical test or performance ranking was run. All scientific
logical-quality and hardware gates remain unpassed.

| Seed | Absolute step | Held-out mean BCE | BCE / step 200 | Brier score | Positive predicted labels | Shots active at 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 200 | 0.046750209 | 1.000000 | 0.002905355 | 0 | 0/4096 |
| 11 | 600 | 0.015318587 | 0.327669 | 0.001302529 | 0 | 0/4096 |
| 11 | 2000 | 0.004389075 | 0.093884 | 0.000828313 | 4004 | 2524/4096 |
| 22 | 200 | 0.085021627 | 1.000000 | 0.007375910 | 0 | 0/4096 |
| 22 | 600 | 0.023264283 | 0.273628 | 0.001524113 | 0 | 0/4096 |
| 22 | 2000 | 0.005932468 | 0.069776 | 0.000915857 | 0 | 0/4096 |
| 33 | 200 | 0.067845599 | 1.000000 | 0.004989786 | 0 | 0/4096 |
| 33 | 600 | 0.019308838 | 0.284600 | 0.001397309 | 0 | 0/4096 |
| 33 | 2000 | 0.005084081 | 0.074936 | 0.000854256 | 5860 | 3095/4096 |

For seeds 11/22/33 the prior-only BCE references on these same shots were
0.008606493 / 0.008581137 / 0.008599313; they did not change across horizons.
All three final checkpoints had lower descriptive BCE than this reference.
The strictly zero-probability reference has infinite BCE because positive labels
occur: raw reports retain `zero_probability_bce: null` with the explicit
`zero_probability_bce_is_infinite: true` flag rather than serializing infinity
or changing the reference probability.

Each seed has 48738304 mechanism labels, including 60346 / 60142 / 60331 positive
labels. These are correlated within 4096 shots, not 48738304 independent samples.
The same held-out stream was replayed at every horizon of a seed, so repeated
horizons do not multiply the number of independent test samples. These streams
are separate from the previous diagnostic and physical-test streams.

Every control at every horizon had zero exact latent, detector and joint H/L
equality counts against the complete sampled latent correction. This is **not**
a 100% final logical-decoding failure: a predecoder is allowed to leave a residual
for the global decoder, which was not evaluated here. Exact equality to the
entire latent vector and final logical success are different endpoints.

## Execution and provenance

Producing source revision: `c273d8f194f4764f4830d44fa76aec41798324e4` (local
content-equivalent commit `129b12c`). Later report/evidence commits are not the
producing source. The source bundle SHA-256 is
`ddf93a3a69dcc986ee8fc4ff8116fd2a9e1c0308e9b7632903b24c84445d5c61`;
the frozen protocol SHA-256 is
`8065763be33a88bca2ea4538a691971e57269fa5e61bd701aae7b5a54413759a`.

The run started at 2026-10-24 20:17 UTC on the authorized A800-172 host, using
CPU only, Python 3.12.13, Torch 2.13.0+cu130 and the prior dependency lock in image
`sha256:9465e1fca0a8c22af1b2ee657c8a6ee7e9f9f8e74d873b0add84b274da58b9e2`.
The isolated container used eight CPUs, 8 GiB memory, one Torch thread, no network
and no GPU exposure. The four-hour limit was not reached; exit code was zero and
the container was automatically removed. This is a `single_device_fast_path`
research run, without scalability claims.

All three audited step-200 model, Adam and generator states were verified before
formal draws or updates. The unchanged 81-parameter MLP, unweighted mean BCE,
Adam learning rate 0.001, batch size 128, full training DEM and its approximation
were retained. Each seed continued its original batch stream for exactly 1800
updates, retaining steps 200/600/2000: 5400 updates and 691200 new training shots
in total. Held-out sampling used exactly 4096 shots per seed, seed +12100000,
chunks of at most 32 and the same stream per horizon. There were zero new
calibration/validation-selection/physical-primary-test shots.

Inputs and producing source were mounted read-only. The private execution
directory is `/tmp/fq-ai-training-study-20261025-c273d8f-preflight2` on A800-172.
Original checkpoints, all nine resulting checkpoints, source bundle, launch lock,
PID, execution metadata and logs remain there and in the local raw archive
`qec-ai-training-response-20261025`. Old 65536-shot test arrays were never mounted.

The first packaging preflight omitted the earlier label-diagnostic protocol
file, causing one file-not-found failure after 40 tests passed. It ended before
formal data collection. That directory and its failure log are retained. The
corrected bundle was verified in a fresh directory: all 41 tests passed in the
matching environment, followed by all three real checkpoint preflights. No
repository model or study implementation changed between these preflights.
There was one formal run, with no restarts, replacement seeds or extra samples.

## Raw evidence and post-run checks

[Raw reports, update logs and audit](evidence/qec_ai_training_response_20261025/)
retain eleven raw JSON files (manifest, summary and nine horizon reports), three
raw JSONL update streams, complete execution log, preflight log and exit markers.
These files are copied byte for byte. Binary checkpoints remain in the private
raw archive; the public audit records their file and model-state fingerprints.
The complete downloaded archive SHA-256 is
`3233fd2e6744cdff7da3030e48a9aeae1d04cc1ded3dd8f9d61f068367bde685`.

Post-run verification, without new sampling or optimizer calls, checked:

- every raw file hash, producing source file hash and protocol hash;
- all three seeds, nine fixed horizons, 5400 consecutive update counters and
  finite training losses, with no omitted intermediate loss records;
- matching latent hashes across horizons, 4096 per-shot BCE values and paired
  differences per report, and their means and population standard deviations;
- complete exact-feature group observation/label counts, fractions and empirical
  entropy, positive/negative loss accounting and control-count relationships;
- exact equality of each step-200 checkpoint's full contents to its original
  archive, all nine model fingerprints and Adam step counters, and unchanged
  canonical H/L/prior buffers;
- the declared per-seed and joint descriptive decision, without selecting the
  lowest-loss horizon or a favorable seed.

The raw paired differences and quantiles are retained for inspection. No
population significance interval or LER estimate is inferred. Exact feature
frequencies remain finite-sample descriptions, not a population Bayes bound.

## User example

From the repository root, inspect the completed fixed-budget outcome:

```python
import json
from pathlib import Path

root = Path("docs/development/evidence/qec_ai_training_response_20261025")
summary = json.loads((root / "summary.json").read_text())
print(summary["consistent_descriptive_training_response"])
print(summary["logical_quality_claim_allowed"])
report = json.loads((root / "seed-11-step-2000.json").read_text())
print(report["active_shots"]["raw_0.5"])
```

Verified output:

```text
True
False
2524
```

The execution CLI remains in `examples/qec/study_training_budget.py`. It requires
the audited archived inputs and matching producing environment and refuses reused
output directories; this report is not an instruction to rerun the completed
study. No public API or old checkpoint schema has changed.

## Limits and next gate

The original 200-step checkpoint scores respond strongly to this fixed larger
budget while threshold-0.5 activation is inconsistent across seeds. This supports
investigating the existing model's calibrated decisions with a separate
prospective protocol; it does not establish the cause of earlier inactivity,
optimizer convergence or an architectural advantage over Ising. No loss or
feature intervention is justified solely by the response designation.

Before any new logical-quality experiment, freeze fresh calibration/validation
and physical-test budgets, precision analysis, all seeds and selection rules.
Retain step-200 controls, global-only and official Ising Fast on common physical
shots. Official pretrained comparisons remain separate from equal-training-budget
architecture studies. Do not reuse old test arrays for tuning or append samples
until significance. The prior Fast result (72 versus 83 failures, paired gate
inconclusive) and all earlier negative outcomes remain unchanged. No performance,
quantization, export, domestic hardware or real-time stage is unlocked.
