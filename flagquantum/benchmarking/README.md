# Benchmarking

Benchmarking provides maintained, reproducible runners and versioned result
payloads for measuring FlagQuantum. It owns runner discovery, argument handling,
environment capture, measurement methodology, aggregation, and claim ceilings.
The implementation under test must use the same supported execution path as a
user; a benchmark-only fast path is not product evidence.

Benchmarking does not choose Runtime policy, implement production kernels,
declare provider capabilities, or turn a smoke test into a hardware or release
claim. Report aggregators validate and summarize supplied measurements; they do
not create evidence for hardware that was not observed.

## Where to start

- `registry.py`: lazy discovery and metadata for maintained runners.
- `__main__.py`: the `flagquantum-benchmark` command dispatcher.
- `contract.py`: small shared JSON identity and atomic-output helpers.
- `environment_probe.py`: reproducible environment capture.
- `*_local.py`: measured single-process workloads.
- `*_scaling.py`: validation and aggregation entry points.
- `*_scaling_report.py`: versioned report construction.

Research plots and one-off comparisons belong under `benchmarks/`, not in this
importable package. A runner may contain an explicit reference implementation
for correctness comparison, but that reference must remain separate from the
measured production path and be named in the output.

## Ten-minute change path

For a small runner or payload change, modify one owner, update its versioned
contract tests, and run:

```bash
python -m pytest tests/unit/test_benchmark_runner_contract.py \
  tests/unit/test_benchmark_runners_cli.py \
  tests/benchmark_contract/test_statevector_local_performance.py \
  tests/benchmark_contract/test_statevector_scaling_report.py \
  tests/benchmark_contract/test_statevector_weak_scaling_report.py \
  tests/benchmark_contract/test_statevector_training_scaling_report.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Keep workload, seed, precision, warmup, sample count, environment, comparison
baseline, distribution semantics, and claim eligibility explicit. CPU or
single-device results must not be promoted to multi-card, multi-node, domestic
accelerator, QPU, or release-grade evidence without the separately required
hardware run and audit.

For differentiable CPU comparisons, use `differentiable_simulator_corpus`. It
measures exact expectation forward time and the full reverse-mode parameter
gradient separately. Its PennyLane comparison uses default.qubit backprop through
the Torch interface so both engines use the same differentiation family. Qiskit
Aer and Cirq remain outside this corpus because their
current FlagQuantum bridges do not provide a matching native Torch-gradient
contract; finite differences are not substituted for a native gradient.

The same runner also provides a separate, method-matched adjoint track. Select
`--engines flagquantum_adjoint flagquantum_adjoint_python_fallback
pennylane_lightning_adjoint` to compare FlagQuantum's fused native reversible
statevector adjoint, the same direct-layout path with its native operator
disabled, and PennyLane Lightning's adjoint. The older
`flagquantum_adjoint_gather_rollback` engine remains available for regression
diagnosis. The `flagquantum_adjoint_forward_cx_rollback` engine keeps the same
native adjoint implementation but disables the reusable-output CPU CX gather,
providing a paired A/B baseline for forward-path changes. FlagQuantum's public entry point is
`hamiltonian.expectation(circuit, differentiation="adjoint")`; this initial
contract supports a batch size of one and real, constant-coefficient Z/ZZ terms.
The optimized single-process CPU path composes adjacent same-qubit gates, applies
whole CX sequences as one permutation, reuses the forward observable diagonal,
and evaluates analytic RX/RY/RZ/RZZ VJPs without materializing a derivative
state. Set `FQ_NATIVE_CPU_ADJOINT=0` to reproduce the Python direct-layout A/B,
or `FQ_STATEVECTOR_ADJOINT_CPU_DIRECT=0` to reproduce the gather-based rollback.
Set `FQ_NATIVE_CPU_CX_GATHER=0` to restore allocating PyTorch `index_select`
for forward CX sequences.
Repeated local CPU adjoint steps retain the parameter-independent Z/ZZ
observable diagonal in a bounded 256 MiB LRU cache. Use
`flagquantum_adjoint_observable_cache_rollback` for paired benchmark runs or set
`FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE=0` for direct rollback.
Native CPU RX/RY/RZ adjoint layers use full-layer, structure-specialized wide
tiles with tile-local gradient accumulation. Adjacent fixed Hadamards use the
same 11-qubit policy while updating ket and adjoint together. Use
`flagquantum_adjoint_rotation_tile_rollback` for paired measurements or set
`FQ_NATIVE_CPU_ADJOINT_WIDE_TILES=0` to restore the legacy 48-gate, two-qubit
rotation tiles and separate fixed-layer state passes.
The rotation reverse sweep directly enumerates zero/one amplitude-pair blocks;
same-qubit RZ/RY/RX Euler triples additionally keep each pair in registers across
all three gradients and inverse rotations. Use
`flagquantum_adjoint_euler_triple_rollback` for paired measurements or set
`FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES=0` to restore the prior per-gate,
bit-tested traversal.
The cached Z/ZZ diagonal is consumed by a fused native observable boundary:
one parallel pass evaluates the expectation and one seeds the reverse sweep.
Use `flagquantum_adjoint_observable_boundary_rollback` for paired measurements
or set `FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY=0` to restore the eager PyTorch
expressions.
`FQ_STATEVECTOR_CPU_DIRECT_LOCAL=0` independently
restores the distributed gather implementation for ordinary one-process CPU
forward execution.

## Refreshing native comparison evidence

When a FlagQuantum optimization changes only the native timing, refresh that
engine without rerunning or rewriting external framework evidence. Pass the
checked-in corpus artifact as both the refresh source and the output target:

```bash
flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector dense_nonlocal_statevector \
  --n-wires 18 22 --engines flagquantum_native \
  --threads 1 --warmup 1 --iterations 9 --calls-per-sample 1 \
  --refresh-from benchmarks/results/comparison/simulator_workload_corpus_cpu_arm64_20260924.json \
  --json-output benchmarks/results/comparison/simulator_workload_corpus_cpu_arm64_20260924.json \
  --markdown-output benchmarks/results/comparison/SIMULATOR_WORKLOAD_CORPUS_CPU_ARM64_20260924.md
```

The refreshed cases may be a subset of the baseline; every case you do not
measure keeps its checked-in payload. Each generated report prints the exact
refresh command for its own artifact, so copy it from the report when refreshing
a different corpus. The refresh fails closed if the workload identity,
methodology, platform, or runtime environment differs from the baseline. It
records refresh provenance, preserves every unmeasured engine payload, and
recomputes derived ratios and the
Markdown table.
