# FlagQuantum Benchmarks

FlagQuantum provides one discoverable command for maintained benchmarks:

```bash
pip install -e .
flagquantum-benchmark list
```

Inspect a benchmark before running it:

```bash
flagquantum-benchmark info statevector_local
```

Every maintained runner accepts its existing benchmark arguments after
`run`. A CPU quick start that does not require accelerator setup is:

```bash
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json

flagquantum-benchmark run statevector_local \
  --device cpu --n-wires 12 --batch-size 4 --layers 2 \
  --warmup 1 --iterations 3 \
  --json-output benchmarks/results/local/statevector_cpu.json

python benchmarks/flagship_mps_training.py \
  --cases dimer:20 --steps 1 --iters 1 --warmup 0 \
  --json-output benchmarks/results/local/mps_smoke.json

python benchmarks/dynamic_trajectory.py \
  --shots 100 1000 --mid-circuit-measurements 1 2 4 \
  --json-output benchmarks/results/smoke/dynamic-trajectory.json
```

Measure whether the silent `fq.train` path avoids per-step CUDA scalar reads:

```bash
python benchmarks/internal/evidence/train_host_sync.py \
  --device cuda:0 --parameters 1024 --steps 2000 --repeats 9 \
  --output benchmarks/results/local/train_host_sync.json
```

Use `--workload quantum_statevector --wires 12 --parameters 24` to validate the
same behavior through a real `fq.Module` statevector training path.

Use `--device cuda` for the local statevector benchmark on a GPU host.
Distributed measurements are launched with the usual `torchrun` environment;
the maintained report builders are also exposed by the same command:

```bash
flagquantum-benchmark run statevector_weak_scaling INPUT... \
  --json-output benchmarks/results/smoke/weak-scaling.json
flagquantum-benchmark run statevector_strong_scaling INPUT... \
  --json-output benchmarks/results/smoke/strong-scaling.json
flagquantum-benchmark run statevector_training_scaling INPUT... \
  --json-output benchmarks/results/smoke/training-scaling.json
```

## Directory contract

| Directory | Stability | Purpose |
| --- | --- | --- |
| `flagquantum/benchmarking/` | Supported | Packaged CLI, registry, result contract, and maintained runners. |
| `benchmarks/runners/` | Maintained | Reproducible hardware and workload entry points. |
| `benchmarks/results/` | Evidence | Structured local, comparison, smoke, and certified scalability results. |
| `benchmarks/internal/evidence/` | Internal | Development-evidence generators required by current contracts. |

Only `local/`, `comparison/`, `smoke/`, and `scalability/` are evidence
classes. Historical experiment families, plotting scripts, and ad-hoc analysis
belong in the external evidence archive. Migrate them out in reviewable batches according to the
[repository governance policy](../docs/development/REPOSITORY_GOVERNANCE.md).

Top-level Python files are implementation modules retained while their
maintained scenarios are adopted by the registry. Their filenames are not a
public interface; user automation should call `flagquantum-benchmark`.

## Result semantics

Use this result layout:

| Directory | Contents |
| --- | --- |
| `benchmarks/results/local/` | Single-device fast-path results. |
| `benchmarks/results/comparison/` | Cross-framework comparison payloads. |
| `benchmarks/results/smoke/` | Development, replicated, or environment smoke checks. |
| `benchmarks/results/scalability/` | Release-gate scalability payloads only. |

Audit result semantics with:

```bash
python -m benchmarks.audit_results --input benchmarks/results
python -m benchmarks.audit_results \
  --input benchmarks/results/scalability --require-scalability
```

Rank-local replicated kernels, data-parallel throughput checks, and smoke tests
must keep `scalability_claim_allowed=false` and must not use
`claim_evidence_type="production_training_benchmark"`. Release payloads also
require capacity and sharded optimizer-update evidence.

Production promotion is a two-step operation: the executor writes measurements
and a raw log, then `tools/seal_runtime_evidence.py` captures the commit,
workload hash, command, device UUIDs, topology, rank mapping, backend, warmup,
iterations, seeds, and log checksum. It signs the immutable envelope using
`FQ_EVIDENCE_SIGNING_KEY`. Candidates remain under
`benchmarks/results/smoke/release_candidates/` until verification succeeds.

Before an eight-GPU release run, validate the environment without printing
secret material:

```bash
python tools/check_release_evidence_environment.py --world-size 8
```
