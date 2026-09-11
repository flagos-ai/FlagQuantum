"""Continuous benchmark catalog and optimization backlog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    layer: str
    target: str
    correctness: str
    primary_metrics: tuple[str, ...]


BENCHMARK_CATALOG = (
    BenchmarkSpec(
        "micro",
        "matrix/application primitives",
        "analytic matrix",
        ("latency", "allocations"),
    ),
    BenchmarkSpec(
        "kernel",
        "PyTorch and JAX GPU kernels",
        "cross-backend parity",
        ("latency", "GPU activity"),
    ),
    BenchmarkSpec(
        "circuit",
        "native Circuit state/expectation",
        "reference state",
        ("latency", "peak memory"),
    ),
    BenchmarkSpec(
        "training_step",
        "forward/backward/optimizer",
        "gradient parity",
        ("step time", "memory slope"),
    ),
    BenchmarkSpec(
        "communication",
        "rank-owned shard collectives",
        "global checksum",
        ("communication fraction", "progress"),
    ),
    BenchmarkSpec(
        "compile_time",
        "IR compile and JAX JIT",
        "compiled IR/hash",
        ("cold compile", "cache hit"),
    ),
    BenchmarkSpec(
        "startup",
        "import/runtime/distributed init",
        "ready preflight",
        ("initialization", "teardown"),
    ),
    BenchmarkSpec(
        "memory",
        "repeated steady-state steps",
        "stable outputs",
        ("allocated slope", "reserved slope"),
    ),
)

OPTIMIZATION_BACKLOG = (
    "gate fusion and fused parameter kernels",
    "state and tensor layout selection",
    "allocation and communication-buffer reuse",
    "batch vectorization and vmap",
    "compile/JIT cache hit rate",
    "communication-compute overlap",
    "vendor and custom complex kernels",
)

__all__ = ["BENCHMARK_CATALOG", "OPTIMIZATION_BACKLOG", "BenchmarkSpec"]
