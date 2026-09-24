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

## Refreshing native comparison evidence

When a FlagQuantum optimization changes only the native timing, refresh that
engine without rerunning or rewriting external framework evidence:

```bash
flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector dense_nonlocal_statevector \
  --n-wires 18 22 --engines flagquantum_native \
  --threads 1 --warmup 1 --iterations 9 --calls-per-sample 1 \
  --refresh-from benchmarks/results/comparison/workload-corpus.json \
  --json-output benchmarks/results/comparison/workload-corpus.json \
  --markdown-output benchmarks/results/comparison/WORKLOAD_CORPUS.md
```

The refresh fails closed if the workload identity, methodology, platform, or
runtime environment differs from the baseline. It records refresh provenance,
preserves every unmeasured engine payload, and recomputes derived ratios and the
Markdown table.
