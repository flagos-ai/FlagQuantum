# FlagQuantum Testing Manual

This is the human-facing testing entry point for FlagQuantum. It explains which
tests to run for daily development, local runtime work, distributed CPU
semantics, real accelerator work, multi-node work, and benchmark release
validation.

The core rule is simple: run the smallest meaningful tier first, then expand by
blast radius. CPU distributed tests prove semantics and fail-closed behavior
only. They do not prove real multi-card capacity expansion and must never be
used as scalability release evidence.

Do not treat an empty marker selection as verification.

## Correctness Certification And No-Progress Policy

`docs/correctness_certification.json` is generated from the operator/lowering
registry. Every supported operator/backend pair must have a versioned generated
case. Distributed implementation changes must run the healthy required local
GPU lane; CPU simulation cannot replace it. One GPU covers local parity, two
GPUs cover mandatory sharded semantics, and scheduled 4–8 GPU jobs cover rank
count, topology, partition and collective behavior. Release certification owns
release-wide
certification; this foundation is not release evidence.

Long-running jobs must report phase, last operation, completed work, memory,
collective state and rank. A no-progress watchdog classifies stalled input,
collective/participant stalls, rank desynchronization, memory growth, and ranks
holding memory without useful work. It must preserve diagnostics, terminate the
stale job, and verify process-group/child cleanup. Explicit compile and
checkpoint budgets prevent false positives during bounded legitimate work.

Release-grade scalability evidence requires a promoted benchmark payload that
passes `fq.require_distributed_scalability(...)` and the benchmark audit commands
for `benchmarks/results/scalability/`.

## Current Runnable Commands

## Quick Start

| Situation | Command | What It Proves | What It Does Not Prove |
| --- | --- | --- | --- |
| Daily development or any issue baseline | `python tools/ci_tier.py pr-default` | Fast smoke/unit health for imports, minimal circuits, autograd, pure planner/audit helpers. | Runtime integration, distributed behavior, performance, or release readiness. |
| Local API/runtime/planner/compiler changes | `python tools/ci_tier.py pr-runtime` | Seeded `integration` coverage for local runtime/API behavior. | Multi-process transport, GPU execution, or scalability claims. |
| Distributed planner/audit/benchmark contract changes | `python tools/ci_tier.py pr-distributed` | CPU distributed semantics, fail-closed gates, benchmark JSON contracts, release-gate validation. | Real multi-GPU or multi-node capacity expansion. |
| Real multi-GPU or accelerator testing | `python tools/ci_tier.py gpu-scheduled` | Accelerator-backed tests selected by `distributed_accel and gpu`. | Multi-node transport or release scalability by itself. |
| Multi-node transport testing | `python tools/ci_tier.py multinode-scheduled` | Multi-node/rank-placement candidates selected by `distributed_multinode`. | Release claims unless produced payloads also pass release validation. |
| Release candidate or promoted benchmark claim | `python tools/ci_tier.py release` | Full non-performance-benchmark pytest plus benchmark root audit and scalability release scan. | Claims outside the audited payload. |

Equivalent direct marker commands are available when a focused local run is
clearer:

## Planned Marker Commands

```bash
python -m pytest -m "smoke or unit" -q
python -m pytest -m "integration" -q
python -m pytest -m "distributed_cpu" -q
python -m pytest -m "distributed_cpu or release_gate or benchmark_contract" -q
python -m pytest -m "benchmark_contract or release_gate" -q
python -m pytest -m "distributed_accel and gpu" -q
python -m pytest -m "distributed_multinode" -q
python -m pytest --markers
```

benchmark generation remains an explicit command and is never implied by a
marker-only test tier.

Focused file-list commands remain useful for review-sized checks and bisects:

```bash
python -m pytest tests/test_native_circuit.py tests/test_backends.py -q
python -m pytest tests/test_distributed_statevector.py tests/test_jax_distributed_plan.py tests/test_distributed_scalability_audit.py -q
python -m pytest tests/benchmark_contract -q
```

## Marker Reference

| Marker | Runtime Environment | Proves | Does Not Prove |
| --- | --- | --- | --- |
| `smoke` | Fast local CPU; no subprocess, GPU, or benchmark run. | Importability, minimal `fq.Circuit`, expectation, and autograd health. | Planner completeness, distributed semantics, performance, or release readiness. |
| `unit` | Fast local CPU pure logic. | IR, planner, audit, metadata, helper, and schema logic in isolation. | Runtime integration, process-group behavior, accelerator execution, or benchmark truth. |
| `integration` | Local single-process developer environment. | Cross-module API/runtime behavior without cluster setup. | Multi-process transport, GPU execution, or scalability claims. |
| `distributed_cpu` | CPU/local distributed simulators, planner/audit checks, or torch distributed semantics that do not require real accelerator capacity. | Rank ownership metadata, sharding intent, fail-closed gates, and development-production semantic parity where applicable. | Real multi-GPU capacity expansion, production transport performance, or release-grade scalability evidence. |
| `distributed_accel` | Explicit accelerator or multi-GPU environment. | Hardware-backed distributed behavior for the covered path. | Multi-node transport unless also marked `distributed_multinode`; release claim unless audit evidence passes. |
| `distributed_multinode` | Explicit multi-node cluster or scheduled/manual transport job. | Node/rank placement and multi-node transport behavior for the covered path. | Release-grade claim unless benchmark payload and release gate pass. |
| `benchmark_contract` | Fast local pytest over benchmark JSON and audit helpers. | Payload shape, required fields, result hygiene, scanner contract, and non-release payload rejection. | Real benchmark performance or hardware behavior. |
| `release_gate` | Fast local pytest over release validators and promoted payloads. | Fail-closed release validation behavior and accepted release payload contract. | That unmeasured paths are scalable; it only validates provided evidence. |
| `scalability` | Release-grade evidence path, usually scheduled or manual. | A specific promoted payload may support a capacity-scaling claim when it passes the release gate. | Generic distributed readiness for all backends or future workloads. |
| `gpu` | CUDA, vendor, or accelerator device. | Device-specific execution for the covered test. | Distributed semantics unless paired with distributed markers. |
| `distributed` | Umbrella marker for distributed planning, runtime, or evidence tests. | The test touches distributed behavior in some form. | Which environment is required; use `distributed_cpu`, `distributed_accel`, or `distributed_multinode` for CI policy. |
| `slow` | Any environment, intentionally slower than default loops. | Longer-running behavior selected explicitly. | Release readiness or scalability on its own. |

## Test Tiers

### Push Gate

Install both repository hooks once per clone:

```bash
pre-commit install
```

The checked-in configuration installs `pre-commit` and `pre-push`. Before a
push leaves the machine, the push gate runs:

- all normal pre-commit quality and source-of-truth checks;
- the strict typed trainable-module and execution-mainline checks;
- capability maturity, required-check policy, and lazy-import validation;
- `pr-default`, `pr-runtime`, and `pr-distributed`.

Run the same gate directly when diagnosing a failure:

```bash
python tools/pre_push.py
python tools/pre_push.py --dry-run
```

The gate stops at the first failure. It is CPU-safe and does not claim to
replace the remote Python-version matrix, clean distribution installs,
supply-chain checks, or accelerator jobs.

### Daily Development

Use daily development for ordinary code edits and as the minimum verification
entry point for every change.

```bash
python tools/ci_tier.py pr-default
```

This runs `smoke or unit`. It must stay fast, local, and free of torchrun, GPU,
benchmark generation, or distributed setup. The baseline must not require torchrun, process groups, benchmark execution,
GPU hardware, or distributed setup.

### Local API And Runtime Work

Use this tier for changes touching `fq.Circuit`, backend selection, runtime
selection, local execution, planner behavior, compiler integration, or other
single-process API boundaries.

```bash
python tools/ci_tier.py pr-runtime
# direct:
python -m pytest -m "integration" -q
```

This tier is seeded by `tests/test_native_circuit.py` and
`tests/test_backends.py` without moving directories.

### Distributed CPU Semantics

Use this tier for distributed statevector plans, JAX distributed preflight,
runtime metadata, audit behavior, fail-closed claimability, and local
development simulators.

```bash
python -m pytest -m "distributed_cpu" -q
python -m pytest -m "distributed_cpu or release_gate or benchmark_contract" -q
```

This layer proves metadata semantics and fail-closed behavior only. It does not
prove real multi-card capacity expansion, production accelerator transport, or
release-grade scalability evidence. It does not prove real multi-GPU or multi-node capacity expansion. Never use CPU distributed tests alone as
scalability release evidence.

### Real Multi-GPU Or Accelerator Work

Use this tier only on an explicit accelerator runner.

```bash
python tools/ci_tier.py gpu-scheduled
# direct:
python -m pytest -m "distributed_accel and gpu" -q
```

A local CPU run may select these tests and skip them; that is wiring validation,
not hardware evidence.

The Phase 5 MPS accelerator boundary test requires at least two local non-CPU
JAX devices. When hardware is available it records device placement,
rank-to-device mapping, world/local/node topology, XLA boundary communication,
an estimated live probe-array footprint, backward time, communication time, and
sharded gradient ownership. It is classified as a measured accelerator
development artifact (`claim_evidence_type="development_smoke"` and
`artifact_classification="measured_accelerator_probe"`) and remains
fail-closed: the boundary probe is not the full MPS backward executor, does not
execute the production optimizer, does not prove capacity expansion, and is
single-node evidence only. A skipped test produces no accelerator evidence.

### Multi-Node Work

Use this tier only in a scheduled/manual torchrun or cluster environment with
explicit rank placement.

```bash
python tools/ci_tier.py multinode-scheduled
# direct:
python -m pytest -m "distributed_multinode" -q
```

Multi-node tests must not become release claims unless their benchmark payloads
also pass the release gate.

### Benchmark Release Validation

Pytest may validate benchmark contracts and promoted payloads, but real
performance benchmark generation remains explicit and outside ordinary pytest
runs. Use these commands before promoting benchmark JSON or release notes:

```bash
python benchmarks/audit_results.py --input benchmarks/results
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

Release-grade scalability requires one logical workload sharded across ranks,
`distribution_semantics="sharded_across_ranks"`, accepted release evidence,
capacity-failure evidence, sharded gradient/optimizer ownership, communication
and memory evidence, topology metadata, and an empty blocker set.

Before running the single-node MPS NCCL certification matrix, execute:

```bash
python tools/check_mps_single_node_certification_environment.py \
  --json-output benchmarks/results/smoke/mps-single-node-preflight.json
```

The preflight requires a clean tree, eight visible GPUs of one model, PyTorch
NCCL support, and `FQ_EVIDENCE_SIGNING_KEY`. It prints the frozen 1/2/4/8-rank
measured-training commands but remains non-claimable plan evidence. A blocked
preflight must not be presented as runtime or scalability certification.

## CI Policy

GPU and multi-node tiers must run on explicitly provisioned environments.
The checked-in `ci.yml` maps this policy to five primary CPU jobs:

- `quality`: Ruff and Black over `flagquantum/`, `tests/`, and `tools/`, plus
  dependency-policy synchronization, architecture-boundary, generated-document,
  capability-maturity, repository-hygiene checks, and
  the currently enforced typed-foundation subset;
- `cpu-core`: Python 3.10-3.12 smoke/unit and integration with core dependencies
  only, including proof that importing and differentiating a native circuit
  does not import JAX;
- `jax-optional`: the JAX extra and its focused hybrid/distributed regression;
- `qiskit-optional`: real Qiskit IR round trips and local Aer compatibility,
  isolated from the core environment;
- `package`: wheel/sdist construction, forbidden-content inspection, and a
  commit/environment/checksum manifest.

`distributed-cpu` runs semantic and release-contract checks separately from the
core matrix. `scheduled-hardware.yml` is the explicit self-hosted GPU and manual
multi-node entrypoint. A skipped hardware test remains wiring information only,
never execution proof.

| CI Tier | Trigger Condition | Command | Blocks Ordinary PRs | Proof Boundary |
| --- | --- | --- | --- | --- |
| PR default | Every pull request and ordinary local issue verification. | `python tools/ci_tier.py pr-default` | Yes | Fast `smoke or unit` health only. |
| PR runtime/API/planner | Pull requests that touch runtime, API, planner, compiler, or integration boundaries. | `python tools/ci_tier.py pr-runtime` | Only when selected for that blast radius. | Seeded `integration` runtime/API files; not distributed or scalability evidence. |
| PR distributed/audit/benchmark | Pull requests that touch distributed planning/runtime metadata, audit, benchmark JSON, or release gates. | `python tools/ci_tier.py pr-distributed` | Only for distributed/audit/benchmark changes. | CPU distributed semantics plus benchmark contract and release-gate validation; not capacity evidence. |
| Nightly CPU | Scheduled nightly CPU-safe validation or explicit maintainer request. | `python tools/ci_tier.py nightly` | No | Broad non-benchmark, non-accelerator, non-multinode health. |
| GPU scheduled | Scheduled/manual job on an explicit accelerator runner. | `python tools/ci_tier.py gpu-scheduled` | No | Accelerator-backed behavior for selected tests; not multi-node or release evidence by itself. |
| Multi-node scheduled/manual | Scheduled/manual torchrun or cluster job with explicit rank placement. | `python tools/ci_tier.py multinode-scheduled` | No | Multi-node transport candidates for the configured cluster; release claims still require audit payloads. |
| Release | Release candidate validation before promoting benchmark payloads or publishing release notes. | `python tools/ci_tier.py release` | Release only | Full non-performance-benchmark pytest coverage plus benchmark audit and scalability release scan. |

## Path Classification Mapping

| Path Classification | Primary Test Tier |
| --- | --- |
| `single_device_fast_path` | `smoke`, `unit`, local `integration` |
| `sharded_across_ranks` planner/audit/preflight | `distributed_cpu`, `release_gate` when validating claim rejection or promotion |
| `rank_local_replicated_kernel` | `distributed_cpu` or `distributed_accel`, but always non-scalability unless release evidence says otherwise |
| `replicated_per_rank` | `distributed_cpu` or `benchmark_contract` for hygiene and rejection |
| `manual_sliced_tensor_contraction` | local `integration`, plus benchmark/audit contract tests when claims are emitted |
| `observable_term_parallel` | local `integration`, plus benchmark/audit contract tests when claims are emitted |
| real multi-GPU production execution | `distributed_accel`, `benchmark_contract`, `release_gate` |
| real multi-node production transport | `distributed_multinode`, `benchmark_contract`, `release_gate` |
| benchmark JSON/schema/audit contract | `benchmark_contract`, `release_gate` |

## Agent Workflow

`AGENTS.md` is the concise agent-facing workflow for this tiered policy.

`AGENTS.md` is the concise agent-facing workflow. It should stay consistent with
this manual: start from `python tools/ci_tier.py pr-default`, then expand by
blast radius. Focused file-list commands are still useful for review-sized
checks, but tier commands are the official vocabulary.

## Non-Negotiable Release Boundary

CPU distributed tests prove semantics and fail-closed behavior only. They do not
prove real multi-card capacity expansion. Never use CPU distributed tests alone as scalability release evidence. Release claims require accelerator or
multi-node evidence plus benchmark audit and release payload validation.
CPU distributed tests are not scalability evidence and must not be used as release-grade scalability evidence.
### Local GPU gates

`.github/workflows/local-gpu.yml` is path-filtered to distributed/runtime
changes. Its one-GPU behavior job and two-GPU correctness/autograd job run on
the `flagquantum-local` self-hosted runner. The two-GPU job is the required
hardware gate for distributed-runtime changes when that runner is available;
CPU tests cannot substitute for it.

Four- and eight-GPU jobs live in `scheduled-hardware.yml`. They are scheduled
or manually dispatched and do not block unrelated documentation or CPU-only
changes. Hardware manifests and raw logs are uploaded even on failure; skips
must carry a reason and are not evidence of success.
## Coverage gate

CI measures the maintained package with the seeded smoke, unit, and integration
tiers. The CPU maintained-runtime floor is 55%, below the measured 56%
baseline. The floor must only move upward. Benchmark tooling and GPU-only
Triton kernels are owned by their dedicated benchmark and hardware gates;
backend-independent runtime, accelerator orchestration, and multi-node control
paths remain visible in the XML report.
