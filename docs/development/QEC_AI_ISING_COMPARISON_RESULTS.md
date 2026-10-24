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

## Fixed-budget exploratory outcome after access

The user subsequently logged in, accepted the repository conditions and granted
access. The authenticated browser downloaded the official Fast file
`ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors` (1826360 bytes),
SHA-256 `22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05`.
Its metadata states Model 1, R=9, FP16, channels [128,128,128,4] and four 3x3x3
kernels. The existing loader promoted these stored weights to FP32 as declared.
The original HTTP 401 was an anonymous-access restriction, not absent weights.

The published protocol and implementation were unchanged during the full run.
All three native seeds trained for 200 steps; all temperatures selected 0.5 and
all raw/calibrated validation thresholds selected 1.0. All three checkpoints and
then all choices were saved before constructing the shared 65536-shot test stream.

| Pipeline | Test logical failures / 65536 | Wins / losses against global-only |
| --- | --- | --- |
| Global-only PyMatching | 528 | reference |
| Native raw, each of seeds 11/22/33 | 528 | 0 / 0 |
| Native calibrated, each of seeds 11/22/33 | 528 | 0 / 0 |
| Official Ising Fast | 533 | 111 / 116 |
| Boundary-mask-only control | 528 | 0 / 0 |

The native calibrated-vs-Ising comparison has 116 wins and 111 losses. Every
declared statistical summary remains **inconclusive**. Ising's repeated records
reuse one frozen model and exactly the same physical shots; they are not three
independent pretrained runs. No samples, seeds or thresholds were added afterward.

The official model card states that distances below R=9 are outside its validated
range. This d=3, CPU, unequal-training-history experiment therefore establishes
the complete loading/sampling/comparison path, **not** a representative ranking,
an Ising regression claim, or native superiority. Native correction remains
inactive. Separate-process batch-one/batch-32 latency, throughput and RSS reports
are retained for inspection, without comparison to published GPU latency claims.

The producing directory is `qec-ai-ising-exploratory-20261024`. Checked-in
[raw records](evidence/qec_ai_ising_exploratory_20261024/results.json), choices,
manifest and four timing reports retain all outcomes. Binary shots/predictions,
native checkpoints and official weights remain local. Full-run fingerprints:

- Original manifest: `ce32721505464b9488b61f3310f57de9d4c7df01ac9058bf78765f5845a4a838`.
- Results: `3879baab4da413a0a1f8cad6dcdea0dd98588cf4c608dd48fe921fce0474a2e8`.
- Choices: `c66bef8657143bd096f91a015adc5069bab48ba2d3f553b5c4495690e196a4c2`.

The archived manifest normalizes upstream provenance after the run: the original
`ising_commit` value is retained as `ising_source.commit`, alongside
`ising_source.repository`. The existing evidence origin table registers this
exact upstream commit as `external_dependency` from `NVIDIA/Ising-Decoding`.
To reconstruct the original manifest, remove `raw_manifest_sha256` and
`ising_source`, then insert `ising_commit` with the retained commit immediately
after `config`. Serialize with two-space indentation and a final newline; its
SHA-256 must equal the original manifest hash above. No experimental value or raw
results/choices/timing file was edited.

## Remaining gates

Official Fast weights are now locally available; missing-access refusal remains
tested for other users. The two CI failures were documentation-registration
omissions: the optional upstream variable was absent from the testing manual,
and the new domain-import example was absent from the examples index. Both were
added and the two failing policy tests passed without changing their assertions.
A subsequent CI run found that the archived manifest introduced an unregistered
revision field and an inline origin unsupported by the generic provenance tests.
The archive now uses the existing `commit` vocabulary and the external dependency
origin table; future manifests emit the same nested source record. No provenance
gate, supported-origin vocabulary or test assertion was changed.

The initial implemented contender is the native MLP. A common-layout native CNN,
matched-budget upstream retraining, larger-distance/noise/basis experiments and
optimized device deployment remain separate future work. They require explicit
prospective budgets and correctness checks. Completion of this integration does
not satisfy the existing scientific benefit gate.

GPU work is assigned to the user's A800-172/A800-171 hosts, not the local CPU
workstation. A new protocol must use a validated distance and define device,
precision, transfers, budgets and timing boundaries before collecting GPU data.
An isolated auto-removed A800-172 container ran a four-element CUDA tensor check
successfully (PyTorch 2.13.0+cu130); this checks availability only. The official
weight file was copied to a private task directory there and its SHA-256 matched
the local file. No credentials or browser session were transferred. A800-171
still needs the matching isolated image; shared containers were not modified.
