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

Hardware jobs publish run-scoped artifacts through CI. A reviewed baseline may
be supplied to `tools/evaluate_performance_artifact.py`; absent baselines never
silently become release evidence. Only current, public claims belong under
`benchmarks/results/`.

The optimization backlog covers gate fusion, layouts, allocation reuse,
vectorization, compile/JIT caching, communication overlap and vendor/custom
complex kernels. A performance change must attach before/after artifacts and
retain its correctness gate.

Implementation histories, failed experiments, and superseded engineering
measurements belong in the external evidence archive. This guide retains only
the practices required to produce reviewable current evidence.
