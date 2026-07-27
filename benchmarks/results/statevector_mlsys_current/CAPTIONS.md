# Current distributed-SV figures

## Figure 0 — Final matched scaling comparison

Matched 31-qubit, depth-8, complex64 adjoint value-and-full-gradient benchmark
using two unmeasured warmups followed by five measured samples. Error bars span
the minimum and maximum measured samples. The header circuit is a schematic of
one repeated layer: trainable RY gates cover all 31 qubits, followed by a
directed nearest-neighbor CNOT chain; eight layers contain 248 trainable
parameters and 488 gates in total. FlagQuantum uses one node through eight GPUs
and two nodes at 16 GPUs. PennyLane Lightning-GPU uses the same GPU placement;
its one-GPU 31-qubit process terminated with signal 11 and is reported as a
failure rather than assigned a runtime. FlagQuantum reaches 5.37× speedup at
eight GPUs and remains within 10.7% of that runtime across two nodes.
PennyLane Lightning-GPU is 1.44–1.52× slower on 2–8 GPUs and 14.68× slower at 16 GPUs.
Panel (c) reports framework-observed phase boundaries rather than equivalent
algorithmic kernels: PennyLane performs its adjoint differentiation inside the
QNode call, leaving only a small Torch backward handoff, whereas FlagQuantum
exposes separate forward and backward executor phases.

## Figure 1 — Current performance

End-to-end value-and-full-gradient runtime for a 31-qubit, depth-8
differentiable state-vector workload on A800 GPUs. (a) FlagQuantum forward and
backward time on one and two nodes. (b) Same-workload comparison against
PennyLane Lightning-GPU on eight GPUs. (c) The runtime–memory trade-off across
successive 16-GPU executor
variants. Lower and further left is better.

## Figure 2 — Optimization journey

Cumulative effect of executor optimizations on the same 31-qubit workload on
eight A800 GPUs. The final executor is 11.1× faster than the original measured
implementation. These bars combine measurements collected during development;
they are an optimization history, not a controlled one-variable-at-a-time
ablation.

## Figure 3 — Multi-node diagnosis and repair

Transport microbenchmarks and the measured 16-GPU repair sequence. Repair
stages are: (1) initial optimized executor, (2) tuned P2P channels, (3) planner
cache, (4) inter-node checkpointing, (5) all-state checkpointing, and (6)
transpose-plus-one-qubit fusion. The channel sweep increases intra-node
bandwidth but does not improve the inter-node link, motivating communication
avoidance rather than further channel tuning.

## Figure 4 — Circuit generality and scaling boundary

Matched 31-qubit, eight-layer differentiable Linear, Ring, and Brickwork
circuits under the same complex64 reversible-adjoint 2+5 protocol. Error bars
span all five measured samples. All three structures strongly scale through
eight same-node A800 GPUs. The two-node 16-GPU points expose the workload-size
boundary: Ring is 19.2% slower than eight GPUs, while Brickwork is non-stationary
and reaches a 25.602 s median. Individual 16-GPU samples and coefficients of
variation are shown rather than hidden by the median.

## Figure 5 — Profiler root cause

An additional post-protocol Torch profiler step, excluded from benchmark
timings, measures rank-level CUDA and NCCL activity for the matched 8-GPU
workloads. Brickwork retains the same effective logical bandwidth as Linear and
Ring (about 46–47 GB/s), but moves 50% more logical bytes and launches 47% more
communication kernels. Its slowdown is therefore attributed to additional
communication work rather than degraded link bandwidth.

## Figure 6 — Scientific training speed and capacity

End-to-end Adam training steps for a differentiable full-width scientific
workload. At 30 qubits, two-node 16-GPU execution accelerates forward, backward,
and end-to-end time by 4.87×, 6.46×, and 5.42× over one A800. A matched 35-qubit
workload is measured to OOM on one 80-GB A800, while 16 A800 GPUs complete
forward, backward, and optimizer update in 114.319 s with 69.4 GB peak memory
per rank.

## Figure 7 — Dependency-DAG scheduling for Brickwork

A communication-aware list scheduler reorders only dependency-independent gates
while preserving the original instruction order along every logical wire.
For the matched 31-qubit, eight-layer Brickwork workload, the optimization
reduces eight-GPU persistent layout swaps from 72 to 3, logical communication
from 154.6 to 6.44 GB, and communication kernels from 448 to 34. Eight-GPU
end-to-end time improves from 6.475 to 4.233 s. At 16 GPUs, time improves from
25.602 to 2.574 s, peak memory falls from 62.3 to 8.0 GB per rank, and
coefficient of variation falls from 31.2% to 0.27%. The optimized
1/2/4/8/16-GPU medians are 28.837/15.685/8.281/4.233/2.574 s.

## Figure 8 — Ring scheduling and compiled execution

The dependency-DAG scheduler halves the eight-GPU Ring layout exchanges and
logical communication, but scheduling alone leaves end-to-end time unchanged
(5.350 to 5.365 s) because it fragments long gate runs and increases launch
work. Compiling the resulting local-CX segments converts the communication
reduction into an end-to-end improvement: the matched 1/2/4/8-GPU medians
change from 29.241/18.114/9.879/5.350 s to
29.241/12.079/7.525/4.516 s. On eight GPUs, logical bytes fall to 50% and
communication kernels to 53% of baseline. All points execute the same
31-qubit, eight-layer, 248-parameter Ring value-and-gradient workload with two
warmups and five measured samples. The same-version 16-GPU rerun is pending and
is deliberately absent.

## Evidence status

These are measured development figures. The same-version Ring/Brickwork matrix,
profiler breakdown, bootstrap report, and matched capacity/OOM evidence are now
included. Formal release promotion and the remaining controlled
one-variable-at-a-time ablations are still required before calling the
evaluation submission-complete.
