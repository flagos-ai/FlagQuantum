# Continuous performance and memory engineering

FlagQuantum measures performance as a correctness-gated architectural property.
The catalog in `benchmarks/performance_catalog.py` covers micro, kernel, circuit,
training-step, communication, compile-time, startup and repeated-step memory
layers. PyTorch and JAX results remain separate, as do local fast-path latency,
distributed throughput and rank-sharded capacity behavior.

Every performance artifact records warmup and steady-state samples, robust and
ordinary variance, initialization/teardown, allocated/reserved peak and slope,
useful-work/idle/communication fractions, heartbeat gaps, completed work units,
correctness and hardware-specific thresholds. Median absolute deviation controls
short-kernel outliers while the ordinary coefficient remains visible.

The initial A100 development baselines are under `benchmarks/development/`:

- one-GPU PyTorch and one-GPU JAX kernels;
- two-GPU distinct rank-owned state shards with NCCL progress;
- four- and eight-GPU topology/communication behavior.

They are development, non-release artifacts. They do not establish a production
scalability claim. Scheduled jobs rerun the same commands and compare latency,
memory, variance, activity and correctness against these hardware-specific
baselines with `tools/evaluate_performance_artifact.py`.

The optimization backlog covers gate fusion, layouts, allocation reuse,
vectorization, compile/JIT caching, communication overlap and vendor/custom
complex kernels. A performance change must attach before/after artifacts and
retain its correctness gate.

The implementation history, engineering measurements, failed experiments, and
next-stage plan for static parameter programs and the SV/MPS/TN backend kernels
are recorded in [STATIC_PROGRAM_OPTIMIZATION_REPORT.md](STATIC_PROGRAM_OPTIMIZATION_REPORT.md).
