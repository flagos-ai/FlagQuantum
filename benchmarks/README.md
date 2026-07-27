# FlagQuantum Benchmarks

## Local Dense Statevector Performance

Run the fixed, same-IR optimized-versus-sequential baseline on CPU:

```bash
python benchmarks/statevector_local_performance.py --device cpu --n-wires 12 --batch-size 4 --layers 2 --warmup 3 --iterations 10 --json-output benchmarks/results/local/statevector_cpu_baseline.json
```

Use `--device cuda` on a GPU host. The payload records timing samples,
coefficient of variation, peak CUDA allocation, numerical error, fusion/runtime
counters, and the sequential dense reference. It is always classified as
`single_device_fast_path` with scalability and release claims disabled.

Measure pair/subgroup exchange pipelining under `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 benchmarks/statevector_exchange_overlap.py --backend nccl --n-wires 20 --layers 4 --exchange-buffer-bytes 4194304 --warmup 2 --repetitions 10 --json-output benchmarks/results/local/statevector_overlap_20q_2xa800_development.json
```

The runner alternates synchronous and pipelined order, aggregates the slowest
rank, records host and CUDA-event samples, and requires the paired 95% confidence
interval to exclude zero before recommending the pipeline. It remains a local
development artifact and cannot satisfy release-certification gates.

## Flagship Single-Machine JAX-MPS

Quick smoke:

```bash
python benchmarks/flagship_mps_training.py --cases dimer:20,hardware:4 --steps 1 --iters 1 --warmup 0
```

Structured 1000q low-bond MPS:

```bash
python benchmarks/flagship_mps_training.py --cases dimer:1000 --steps 100 --iters 10 --warmup 3 --jax-cache-dir .fq_jax_cache --json-output benchmarks/results/local/flagship_mps_1000q_dimer_cpu_jax.json
```

Hardware-efficient low-bond MPS scale:

```bash
python benchmarks/flagship_mps_training.py --cases hardware:60,hardware:300,hardware:600,hardware:1000 --layers 2 --max-bond 4 --steps 0 --iters 3 --warmup 1 --jax-cache-dir .fq_jax_cache --json-output benchmarks/results/local/flagship_mps_hardware_scale_cpu_jax.json
```

Render a Markdown report:

```bash
python benchmarks/report_flagship_mps.py --input benchmarks/results/local/flagship_mps_1000q_dimer_cpu_jax.json --output benchmarks/results/local/flagship_mps_1000q_dimer_cpu_jax.md
```

Audit result semantics:

```bash
python benchmarks/audit_results.py --input benchmarks/results
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

These flagship MPS benchmarks are `single_device_fast_path` evidence. They do
not claim distributed sharded scalability and do not claim arbitrary
high-entanglement 1000-qubit circuit training.

Use this result layout:

| Directory | Contents |
| --- | --- |
| `benchmarks/results/local/` | Single-device fast-path results. |
| `benchmarks/results/comparison/` | Cross-framework comparison payloads. |
| `benchmarks/results/smoke/` | Development, replicated, or environment smoke checks. |
| `benchmarks/results/scalability/` | Release-gate scalability payloads only. |

Use `--require-scalability` only for `benchmarks/results/scalability/`.
Rank-local replicated kernels, data-parallel throughput checks, and smoke tests
must keep `scalability_claim_allowed=false` and must not use
`claim_evidence_type="production_training_benchmark"`. A release payload also
needs `single_gpu_expected_oom=true`, capacity baseline/failure details, and
sharded optimizer-update evidence.

Production promotion is a two-step operation: the executor writes measurements
and a raw log, then `tools/seal_runtime_evidence.py` captures commit, workload
hash, command, device UUIDs, topology, rank mapping, backend, warmup,
iterations, seeds and log checksum and signs the immutable envelope using
`FQ_EVIDENCE_SIGNING_KEY`. The release scanner verifies that signature before
evaluating scalability semantics. Plans and development runs cannot populate
runtime-measured fields.

Candidates remain in `benchmarks/results/smoke/release_candidates/` until that
process succeeds. Never place unsigned measurements or raw capacity comparison
JSON directly in `benchmarks/results/scalability/`.

Before starting an eight-GPU release run, validate the environment without
printing secret material:

```bash
python tools/check_release_evidence_environment.py --world-size 8
```
# Layout

Executable, reproducible drivers are being consolidated under
`benchmarks/runners/`; exploratory sweeps and plotting belong under
`benchmarks/research/`. Existing top-level scripts are compatibility entry
points until their contracts are covered by tests and migrated safely.
