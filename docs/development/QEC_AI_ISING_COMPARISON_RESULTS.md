# Ising comparison integration outcome

This is software integration evidence, not a pretrained-model ranking or a new
logical-error-rate result. The prospective [protocol](QEC_AI_ISING_COMPARISON.md)
defines the bounded exploratory experiment and the refusal paths.

## Completed prerequisites

- The native private MLP can opt into complete visible hyperedges without
  decomposing their labels. Default graphlike refusal, public APIs and checkpoint
  version/state names remain unchanged. Its extracted residual/frame operation
  is shared by existing single-shot decoding and the batched comparison.
- The executable loads the pinned upstream Model 1 architecture and full
  detector-to-residual/logical-frame pipeline. Its real SafeTensors loading path
  was tested with a clearly untrained local fixture; parameter parity and file
  fingerprint checks passed. This fixture is not an official pretrained model.
- Original-circuit detector samples, observables, common uncorrelated PyMatching,
  raw/calibrated native controls, direct native-vs-Ising paired counts and
  separate-process CPU latency/throughput/RSS paths executed successfully.
- Both X- and Z-memory no-correction checks establish exact detector/frame parity
  with the upstream's explicit terminal complement mask. The upstream source and
  its oracle test describe those masked channels. A separate mask-only control
  prevents attributing preprocessing effects to learned correction.

The integration cell has 24 detectors and 178 independent training-DEM mechanisms:
20 degree-one, 65 degree-two, 60 degree-three and 33 degree-four signatures. All
93 higher-degree signatures are retained. Training/matching use an explicit
disjoint-noise approximation; validation/test circuit sampling does not.

## Verification

The focused test scope includes the new comparison, existing MLP/CNN training,
latent-label checks, paired assessment and calibration tests. The optional
upstream tests were actually enabled with `FLAGQUANTUM_ISING_SOURCE`, not counted
as passed skips. Strict mypy covers the two touched implementation modules, the
example and its tests; formatting/lint and architecture/language checks also ran.
The exact final counts and commands are recorded in the PR description.

The repository default tier was run with a 120-second process-group bound. It
timed out during the existing QEC suite at approximately 8%; this is incomplete
verification, not a passing default tier. Full-history GitHub CI must run on the
published PR head. No CI threshold or workflow was changed.

The final standalone smoke used Python 3.11.16, PyTorch 2.10.0, CPU FP32 and one
Torch intra-op/inter-op thread. It completed four paths, 32 shared test shots,
two native steps and batch-one/batch-32 timing workers. Its Ising weights were
seeded and **untrained**. The summary sets `scientific_ranking_allowed=false`.

Local evidence directory: `qec-ai-ising-comparison-final-20261024` in the producing
workspace's parent. It contains the circuit, DEM, manifest, choices, checkpoints,
test shots, predictions and four worker reports. Fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `c009c57b7b4e1a5d56c56ecc9dcdfead7334e4a85bed5945b3e16d88ddb64022` |
| `results.json` | `dbfa86fec01f65cdacd914e00dd64cf8a0194664b13aaaa10bc2dcc8f60f7fb5` |
| `choices.json` | `1fc1b49af804de45ac6e474a6aee50d12721c3bbc5d0b04e02864e84b5350bac` |
| `circuit.stim` | `112f3cf511686ff35f98c04f014cb413e2e74096b76b76bc480733e99204676c` |
| `training.dem` | `e8a122053fa82fcabc7bf497966e9a6a1fe62a499bb969eff53536da1acd1096` |

Weights and raw Torch archives remain local; the committed command regenerates
this software check. Smoke latency/accuracy numbers are not promoted as research
comparison evidence. Original assessment and calibration evidence is unchanged.

## Remaining gates

Official Fast weight access returned HTTP 401 `GatedRepo`; the user has no local
checkpoint. The non-smoke command therefore correctly refuses before sampling.
This external access requirement is separate from implemented loading support.
Do not repeatedly retry access or replace it with random-model scores.

The initial implemented contender is the native MLP. A common-layout native CNN,
matched-budget upstream retraining, larger-distance/noise/basis experiments and
optimized device deployment remain separate future work. They require explicit
prospective budgets and correctness checks. Completion of this integration does
not satisfy the existing scientific benefit gate.
