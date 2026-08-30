# Submission Figure Captions

## Figure 1 — Distributed training strong scaling

**Strong scaling of differentiable exact-statevector training on one
eight-A800 node.** The fixed 28-qubit Full-width Linear HEA uses complex64
amplitudes, RY rotations, directed CX entanglers, explicit sharded adjoint
backward, and owner-sharded Adam. (a) Median end-to-end and backward time.
(b) Speedup relative to one GPU; the dashed line denotes ideal linear scaling.
(c) End-to-end parallel efficiency and measured per-rank peak allocation,
both normalized to one GPU. Eight GPUs achieve 6.70x end-to-end and 6.17x
backward speedup while reducing per-rank peak allocation to 30.8% of the
one-GPU value. All distributed points match the one-GPU raw gradients within
5.96e-8.

## Figure 2 — Optimization mechanisms

**Ablations of compute, communication, topology, and optimizer
synchronization.** (a) A FlagQuantum-owned fused Triton VJP improves
two-GPU backward by 1.79x. (b) Direct-address packing for cross-shard CX
reduces Ring-workload backward communication by 42.9%. (c) Topology-aware
rank-bit ordering reduces per-rank inter-node backward traffic by 22.2% on
two nodes and 16 A800 GPUs. (d) Owner-masked packed all-reduce replaces 192
per-parameter broadcasts with one collective and reduces Adam plus parameter
synchronization from 15.35 ms to 9.67 ms. Each A/B pair uses identical
workloads and preserves raw-gradient or updated-parameter correctness.

## Figure 3 — End-to-end training and capacity expansion

**Multi-node acceleration and capacity expansion for scientific training
workloads.** (a) A 30-qubit, 59-gate Full-width Linear HEA training step
decreases from 23.45 s on one A800 to 4.32 s on 16 A800 GPUs across two
nodes. (b) The resulting speedups are 5.42x end-to-end and 6.46x for
backward; distributed value and raw-gradient errors are 5.96e-8 and
2.98e-8. (c) The matched 35-qubit workload triggers a measured CUDA OOM on
one 79.25-GiB A800 when requesting a 256-GiB state allocation, while the
16-GPU execution completes multiple training steps with 64.61 GiB peak
allocation per rank and without full-state reconstruction.

## Figure 4 — Statistical significance and external reference

**Long-sample training evidence and an isolated external forward
reference.** The frozen 24-qubit, depth-8 Full-width Linear HEA contains
376 gates and 192 trainable parameters. One- and eight-GPU measurements use
five warmups followed by 30 device-synchronized samples. (a) Independent
bootstrap median-ratio estimates with 95% confidence intervals; end-to-end
speedup is 1.644x [1.636, 1.653]. (b) The complete end-to-end sample
distributions. Cross-world value and raw-gradient errors are 1.12e-8 and
2.24e-8. (c) On the same A800, exact complex64 20-qubit RY/RZ/CX forward
execution measures 2.185 ms for FlagQuantum and 1.259 ms for NVIDIA
cuStateVec, with 1.33e-7 maximum final-state error. cuStateVec is an isolated
benchmark-only reference and is not a FlagQuantum dependency.

## Figure 5 — Matched differentiable external baseline

**External baselines with separated semantics.** (a) The raw 20-qubit forward
comparison measures 2.185 ms for FlagQuantum and 1.259 ms for NVIDIA
cuStateVec. (b) The matched 24-qubit, depth-8 Full-width Linear workload
contains 376 gates and 192 parameters; expectation value plus the complete
adjoint gradient takes 2.376 s in FlagQuantum and 0.288 s in PennyLane
Lightning-GPU, an 8.26x gap. Value and maximum-gradient errors are 1.86e-8
and 2.24e-8. Lightning-GPU is a cuQuantum-backed external system; neither it
nor cuQuantum is a FlagQuantum runtime dependency.

## Evidence and scope notes

- Hardware: NVIDIA A800-SXM4-80GB.
- Single-node scale: one host with up to eight GPUs.
- Multi-node scale: two hosts with eight GPUs each.
- Precision: complex64.
- Figure 1 points are retained matched development measurements from the
  optimization campaign; Figure 4 is the strict long-sample statistical point.
- Both external comparisons are single-device. Only the Lightning-GPU
  comparison includes the complete gradient; neither is a distributed
  training comparison.
- The 256-GiB bar is the failed allocation request, not steady-state resident
  memory.
- These plots are derived development evidence and do not claim a signed
  release artifact.
