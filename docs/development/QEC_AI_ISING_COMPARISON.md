# Ising comparison: functional prerequisite and prospective CPU protocol

Owner: Core. Date: 2026-10-24. This protocol precedes comparison data collection.
It preserves the inconclusive assessment and calibration studies, including their
raw evidence. No public API, old checkpoint schema or production dependency changes.

## Functional prerequisites

Use NVIDIA Ising-Decoding commit
`ea7c1bb3b823fc6a38472d0d95974b6d01932288`, not a reimplementation of its CNN.
The executable checks every Python source under `code/` and every `conf/*.yaml`:
sorted relative path, NUL, file bytes, NUL, concatenated and SHA-256 hashed.
Expected digest: `ed9e6c56f263e0cd9fcc04a494e9cf6544c9f78cbb54c788fcce71b2e961b70e`.
Modified sources fail before sampling. No source is vendored into FlagQuantum.

The native local MLP has an explicit private opt-in for visible hyperedges.
One mechanism remains one label and flips its entire detector/logical signature
together. The default graphlike restriction remains. Exclusive-location groups,
invisible mechanisms and unsupported probabilities still fail. This is a CPU
reference, not a new hypergraph global decoder or a physics correctness claim.
`correct()` exposes the existing residual/frame operation; `decode()` reuses it.
Both contenders use the same uncorrelated PyMatching `decode_batch` call.

The benchmark constructs Ising's boundary-inclusive XV surface memory with its
explicit 25-parameter `NoiseModel.from_single_p` noise. Validation and testing
sample that original Stim circuit. The matcher uses Stim's decomposed DEM. Native
training uses the undecomposed DEM, retaining hyperedges. Both DEM constructions
explicitly set `approximate_disjoint_errors=True`: categorical Pauli alternatives
are approximated as independent priors for training/matching, not for test-shot
generation. The report exposes this approximation and the exact noise parameters.
This experiment does not claim equality with native phenomenological memories.
It does not replace the separately developed general Stim experiment bridge.

The pinned upstream masks the terminal complement-stabilizer input: Z at the
last round in X memory, X at the last round in Z memory. Its oracle tests also
explicitly exclude those masked channels. With all-negative logits the upstream
pipeline therefore does **not** preserve every input detector bit. Keep that
original behavior, record a separate boundary-mask-only global-decoder control,
and test exact no-correction parity on every other detector and the logical frame
in both bases. Do not silently repair upstream or attribute this mask's effect
to learned correction. A shared physical input and matcher do not imply identical
internal preprocessing. This limitation is part of the comparison.

## Fixed exploratory experiment

This initial CPU comparison is exploratory, not a confirmatory ranking or a
resource-matched training contest. Official pretrained training costs are unequal
and incompletely known. It covers one cell only: distance 3, 3 rounds, basis X,
rotation XV, p=0.003; no generalization or large-distance claims are allowed.

| Stage | Fixed budget |
| --- | --- |
| Native seeds | 11, 22, 33; local MLP; keep every seed |
| Training | 200 Adam steps, batch 128, lr=0.001; seed+104729 |
| Calibration | 4096 independent-DEM shots; seed+4000000; existing temperature grid/tie rule |
| Validation | 4096 original-circuit shots; seed+5000000 |
| Native thresholds | 0.5, 0.7, 0.9, 0.99, 1.0; minimum logical failures, tie larger |
| Ising thresholds | Published deterministic logit 0.0 for data and syndrome; no test tuning |
| Test | 65536 original-circuit shots, seed 6000011, shared across all paths/seeds |
| Timing | Separate original-circuit stream, seed 7000011; batches 1 and 32; warmup 3, repetitions 32 |

Freeze all training checkpoints, then all calibration/validation choices, before
constructing any primary test sampler. Repeated Ising records share predictions
and samples; they are not independent replications. Record native raw-temperature
and calibrated controls, Ising and global-only paths; compare native calibrated
against Ising directly as well as against global-only. Reuse paired counts,
intervals and exact discordant-pair tests. Report alpha=0.05/12 summaries (three
seeds times four comparisons); these do not promote the prior benefit gate.
No optional stopping, seed selection or test-budget extension is allowed.

All paths run POSIX CPU PyTorch FP32, one Torch intra-op/inter-op thread, inference mode, without JIT or
quantization. An FP16 checkpoint is promoted to FP32; its original rounding is
retained and its hash/metadata recorded. Timings include input conversions, model
forward, residual update, PyMatching and logical XOR. Sampling, graph/model setup,
training and file I/O are excluded. Report p50/p95/p99 complete-batch latency and
throughput; dividing block time by rounds does not establish feedback latency.
Each timing path runs in a fresh bounded process; RSS is the worker peak including
imports, active model and graph, not a tensor-only allocator peak. CPU timings
do not establish TensorRT/GPU deployment competitiveness.

## Run and refusal paths

Install the upstream CPU inference dependencies into an isolated environment;
they remain optional benchmark dependencies. Provide an unmodified pinned source
checkout. Use a new output directory and run from the FlagQuantum repository root:

```bash
python -m pip install numpy torch stim pymatching matplotlib omegaconf safetensors
PYTHONPATH=. python examples/qec/compare_ising.py \
  --ising-source /path/to/Ising-Decoding --smoke \
  --output-directory /tmp/qec-ising-smoke
```

Expected summary: `purpose="smoke"`, four named paths, 32 distinct test shots,
`scientific_ranking_allowed=false`. This executes the real upstream Model 1 with
seeded **untrained** weights, two native training steps and four timing repetitions.
It validates integration only. It cannot stand in for the pretrained baseline.

```bash
PYTHONPATH=. python examples/qec/compare_ising.py \
  --ising-source /path/to/Ising-Decoding \
  --ising-checkpoint /path/to/authorized-surface-model.safetensors \
  --output-directory /tmp/qec-ising-exploratory
```

The second command requires an authorized surface Fast/Accurate SafeTensors file.
Without one it fails before sampling or creating output. Anonymous access to the
official Fast weights returned HTTP 401 `GatedRepo` on 2026-10-24. No credentials,
gated weights or fabricated pretrained scores are committed. A user-supplied file
is hashed; the tool does not certify its publisher or license provenance.

Remaining comparison stages require separate prospective protocols: common-layout
native CNN, matched-budget retraining, distance/noise/basis sweeps, and optimized
device-specific inference. Complete input/label/oracle checks before those runs.
Measure before optimizing; no accelerator adaptation is justified by smoke or
this single-cell exploratory comparison.
