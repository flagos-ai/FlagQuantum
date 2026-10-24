# Prospective A800 Ising comparison

This protocol is committed before new CUDA correctness probes, training or test
collection. It adds an experimental single-device measurement lane, not an
optimization, production promotion or a replacement for the frozen CPU study.
Public APIs, CPU defaults and checkpoint contents remain unchanged.

## Workload and controls

Reuse the pinned upstream and source digest in [the CPU protocol](QEC_AI_ISING_COMPARISON.md).
Require the authorized Fast checkpoint with SHA-256
`22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05`.
Run distance 9, 9 rounds, X memory, XV rotation and p=0.003 using upstream's
complete `NoiseModel.from_single_p` parameters. This distance meets Fast's stated
minimum; it does not establish generalization across distances, bases or noise.
Preserve the upstream terminal complement mask and its mask-only control.
Training/matching use the disclosed independent approximation of disjoint Pauli
alternatives; validation and test samples come from the original circuit.
Flattened duplicate complete detector/logical signatures are canonicalized with
the existing DEM `merge_duplicate_mechanisms()` independent-parity rule. This
preserves the declared independent distribution; detector-identical signatures
with different logical contributions remain separate.

Keep global-only PyMatching, native raw MLP, native calibrated MLP and actual
pretrained Fast paths. All use the same uncorrelated batched CPU PyMatching.
The native geometric CNN is not transplanted onto an incompatible layout.
Native training stays on CPU using the existing trainer. Official training
history and compute differ: this compares deployed checkpoints, not architecture
quality under equal training budgets. No scientific benefit gate is waived.

## Fixed budgets and ordering

| Stage | Budget |
| --- | --- |
| Native training | Seeds 11/22/33; 200 Adam steps, batch 128, lr 0.001; generator seed+104729 |
| Calibration | 4096 independent-DEM shots per seed; seed+4100000; existing temperature grid and tie rule |
| Validation | 4096 original-circuit shots per seed; seed+5100000; thresholds 0.5/0.7/0.9/0.99/1.0; fewest failures, ties choose larger |
| Device correctness | After all choices freeze, before test sampling: 64 circuit shots at seed 8100011, all one-hot detector inputs, all-zero/all-one rows; batches at most 32 |
| Primary test | 65536 shared original-circuit shots, seed 6001011; inference batches 32 |
| Timing inputs | Separate circuit stream, seed 7100011; batches 1/32/128 |
| Timing budget | 10 warmups, 200 measured complete batches per path and batch size; native seed 11 only |

Smoke keeps the same d=9/R=9 device boundary, uses seed 11, two training steps
at batch 16, 32 held-out/test shots, and 3 warmups/4 timing repetitions. It still
requires the actual pinned Fast checkpoint; it is never a performance ranking.

Freeze every training checkpoint, then every calibration/validation decision,
before constructing the primary test sampler. Check exact CPU/CUDA residual and
logical-frame equality on every probe for each selected path and seed. Check
final-prediction equality on the 64 physical circuit shots: arbitrary synthetic
one-hot/all-one detector inputs need not admit a perfect matching. This distinction
is declared during implementation review, before any CUDA probe data collection.
Any mismatch or nonfinite output aborts before primary test collection. No
threshold adjustment to pass parity, silent CPU fallback, seed omission, sample
extension or optional stopping is allowed. Retain aborted evidence separately.
Use alpha=0.05/12 paired summaries (3 seeds x 4 comparisons), including direct
native-calibrated versus Fast. Repeated Fast predictions and training seeds do
not multiply the number of independent physical shots. Insufficient evidence
is inconclusive; a small p-value alone does not satisfy the existing benefit gate.

## Device and measurement boundary

Use one currently idle A800-SXM4-80GB on A800-172 in an isolated auto-removed
container. Record device UUID/model, driver, runtime, PyTorch/dependency versions,
image ID, CPU/thread settings, source/protocol/weight hashes and available memory.
Use FP32 for both contenders, disable TF32 and autocast, enable deterministic
algorithms, and record these settings. Fast's stored FP16 weights are promoted
to FP32. No TensorRT, compilation, quantization, overlap or distributed setup.
All comparisons on a host use the same physical device and environment.

Wall-clock timing starts from CPU binary detector input and ends after CPU
logical prediction. Include H2D, layout conversion, forward, residual/frame
construction, D2H, binary validation, CPU matching and logical XOR. Synchronize
the selected CUDA device before and after each neural batch. Global-only remains
a CPU control with no unnecessary H2D transfer. Exclude sampling, graph/model
construction, training and file I/O. Report complete-batch p50/p95/p99 and
shots/second; do not divide by rounds to imply real-time feedback latency.
Run each timing path in a fresh bounded process, retaining process peak RSS and
CUDA peak allocated/reserved bytes including the active model and DEM buffers.
Neither CUDA allocator peak nor process RSS equals total device memory usage.

## Independent host replication and acceptance

Prepare A800-171 with the exact same isolated image and optional dependencies.
Run the same frozen budgets and random streams once there; do not pool its
identical physical samples as additional evidence. Compare circuit, detector,
truth, choices and prediction hashes/counts across hosts. Report any divergence;
do not tune or rerun until it disappears. Measure latency separately on each
host, with the same contenders/device settings within that host.

Accept this engineering slice only with CPU regression checks, actual d=9 CUDA
bit-parity checks, both complete host runs or an explicit reproducible blocker,
all seeds/controls retained, bounded timing workers and reviewable CI. Report
negative and inconclusive quality outcomes. Architecture superiority, release
latency and production feedback claims remain outside this protocol.

## Run

Use a fresh isolated container exposing exactly one idle physical GPU, with
the repository, pinned upstream checkout, authorized weights and optional
dependencies available. From the repository root:

```bash
PYTHONPATH=.:/task/deps python examples/qec/compare_ising.py \
  --a800 --smoke --image-id "$IMAGE_ID" \
  --ising-source /task/Ising-Decoding \
  --ising-checkpoint /task/fast.safetensors \
  --output-directory /task/smoke
```

`IMAGE_ID` is the full `sha256:...` value from `docker image inspect`, not a tag.
Expect `purpose="smoke"`, four paths, 32 distinct test shots and
`scientific_ranking_allowed=false`. Remove `--smoke` for the fixed full budget
and use a different output directory. Always retain the manifest, parity record,
choices, checkpoints, primary shots/predictions, counts and timing reports.
Missing CUDA, weights, pinned source or required metadata fails before training
or output creation. Never put model weights or account credentials in Git.
