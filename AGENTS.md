# FlagQuantum Agent Operating Manual

This repository is being built into a production-grade quantum AI framework.
Every agent working here must optimize for a coherent FlagQuantum product, not
for isolated demos.

## Product Mission

FlagQuantum must become a flagship quantum AI framework with:

- one public identity: `import flagquantum as fq`
- one source of truth: FlagQuantum IR
- PyTorch as the primary training interface
- JAX as an optional quantum kernel accelerator
- DLPack and `torch.autograd.Function` for cross-framework gradients
- true distributed statevector, MPS, and tensor-network execution
- trainable sharded gradients, not forward-only showcases
- deployment of trained parameterized circuits to quantum cloud or hardware
- first-class CPU, single-GPU, and single-node developer experience
- production multi-GPU and multi-node scale-out without changing user code

## Non-Negotiable Rules

1. Do not describe replicated per-rank execution as distributed scalability.
   A scalability claim is allowed only when one logical workload is partitioned
   across ranks and reported as `distribution_semantics="sharded_across_ranks"`.

2. Do not let distributed orchestration slow down local users. CPU, single-GPU,
   and single-device JAX paths are first-class fast paths.

3. Do not silently fall back from MPS or tensor network to statevector for
   performance, capacity, or gradient claims. If a fallback is used for
   correctness inspection, report it explicitly.

4. Forward-only sharding is incomplete for quantum AI training. Production
   training claims require gradients and optimizer updates to preserve the same
   distribution semantics as forward execution.

5. Every distributed result or benchmark must expose:
   `world_size`, `local_world_size`, `node_count`, rank ownership, memory,
   communication, `distribution_semantics`, `scalability_claim_allowed`, and
   blockers.

6. Keep the user API simple. Advanced controls may exist, but normal usage must
   remain centered on `fq.Circuit`, `fq.Module`, `run`, `plan`, training loops,
   and deployment packages.

7. The repository remains FlagQuantum. Do not introduce public TensorCircuit-NG
   or `tc` branding in user-facing APIs, docs, examples, or benchmark claims.

## Source Documents

Before changing distributed runtime, planners, benchmark claims, or quantum AI
training paths, read:

- `docs/concepts/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md`
- `docs/concepts/DISTRIBUTED_SCALABILITY_PRINCIPLES.md`
- `docs/roadmap/CAPABILITY_MATURITY.md`
- `docs/reference/KNOWN_LIMITATIONS.md`

These documents are binding design standards.

## Architecture North Star

The intended product architecture is:

```text
fq.Circuit / fq.Module
        |
        v
FlagQuantum IR
        |
        +-- compiler / drawer / cloud deployment
        +-- local CPU/GPU PyTorch native runtimes
        +-- JAX quantum kernels exposed through PyTorch autograd
        +-- distributed statevector amplitude sharding
        +-- distributed MPS site/bond sharding
        +-- distributed TN slice/reduction and reverse contraction
```

The same circuit and training code should run in local development and
production. Backend choice is controlled by environment and planner policy, not
by rewriting user code.

## Required Development Workflow

1. Inspect the existing implementation with `rg` before editing.
2. Classify the path being touched:
   `single_device_fast_path`, `sharded_across_ranks`,
   `rank_local_replicated_kernel`, `manual_sliced_tensor_contraction`,
   `observable_term_parallel`, `data_parallel_replicated`, or
   `replicated_per_rank`.
3. Implement or update runtime metadata before writing benchmark language.
4. Add focused tests that fail if semantics are misreported.
5. Preserve existing public APIs unless a migration is explicitly planned.
6. Run the smallest meaningful tests first, then broader tests when the blast
   radius is larger.
7. Update the capability matrix and known limitations when a support boundary
   changes.

## Verification Standards

Use the tiered pytest policy in `docs/development/TESTING.md`: run the smallest meaningful
layer first, then expand by blast radius. Do not start with the largest suite
when a smaller seeded tier proves the touched surface.

Default verification for every change:

```bash
python tools/ci_tier.py pr-default
# equivalent direct command:
python -m pytest -m "smoke or unit" -q
```

Expand by blast radius:

| Change Surface | Primary Tier | Command |
| --- | --- | --- |
| local API, runtime, planner, compiler, or single-device behavior | `integration` | `python tools/ci_tier.py pr-runtime` |
| distributed planning, runtime metadata, audit logic, benchmark JSON, or release-gate code | `distributed_cpu`, `benchmark_contract`, `release_gate` | `python tools/ci_tier.py pr-distributed` |
| accelerator-backed distributed behavior | `distributed_accel`, `gpu` | `python tools/ci_tier.py gpu-scheduled` |
| multi-node transport or rank-placement behavior | `distributed_multinode` | `python tools/ci_tier.py multinode-scheduled` |
| release candidate or promoted benchmark claim | full non-scalability pytest plus benchmark audit | `python tools/ci_tier.py release` |

Only run marker-selection commands that have seeded tests; an empty selection is not verification evidence.

Direct marker commands are also valid when a focused run is clearer:

```bash
python -m pytest -m "integration" -q
python -m pytest -m "distributed_cpu" -q
python -m pytest -m "distributed_cpu or release_gate or benchmark_contract" -q
python -m pytest -m "distributed_accel and gpu" -q
python -m pytest -m "distributed_multinode" -q
python -m pytest -m "benchmark_contract or release_gate" -q
```

Path classification maps to test tiers as follows:

| Path Classification | Required Verification Tier |
| --- | --- |
| `single_device_fast_path` | `smoke`, `unit`, and local `integration` |
| `sharded_across_ranks` planner, preflight, audit, or metadata | `distributed_cpu` plus `release_gate` when claim rejection or promotion is touched |
| `rank_local_replicated_kernel` or `replicated_per_rank` | `distributed_cpu` or `benchmark_contract` proving non-release classification |
| `manual_sliced_tensor_contraction` or `observable_term_parallel` | relevant local `integration` plus benchmark/audit contract tests when claims are emitted |
| real accelerator distributed execution | `distributed_accel` and `gpu`, plus benchmark audit before any public claim |
| real multi-node production transport | `distributed_multinode`, plus benchmark audit and release payload validation |

Current focused file-list commands remain useful for review-sized checks and
legacy bisects:

```bash
python -m pytest tests/test_native_circuit.py tests/test_backends.py -q
python -m pytest tests/test_distributed_statevector.py tests/test_jax_distributed_plan.py tests/test_distributed_scalability_audit.py -q
python -m pytest tests/benchmark_contract -q
```

CPU distributed tests prove semantics and fail-closed behavior only. They may
also check metadata shape and development parity. They do not prove real
multi-GPU or multi-node capacity expansion and must never be cited as
scalability release evidence.
Never use CPU distributed tests alone as scalability release evidence.

Promoted benchmark JSON still requires fail-closed audit validation:

```bash
python benchmarks/audit_results.py --input benchmarks/results
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

Only promote a benchmark as scalability evidence if the audit passes and the
payload demonstrates one logical workload sharded across ranks with release-gate
claim evidence. Real multi-GPU or multi-node production claims require
`distributed_accel` or `distributed_multinode` coverage plus benchmark audit and
release payload validation.

## Benchmark Honesty

Benchmark conclusions must separate:

- local peak performance
- JAX kernel acceleration
- data-parallel throughput
- rank-local replicated execution
- observable-term parallelism
- true capacity expansion through sharding

If PennyLane, TensorCircuit-NG, CUDA-Q, cuQuantum, Qiskit, or another framework
is used as a comparison, the comparison must state backend, device, interface,
gradient method, dtype, warmup, iterations, timeout behavior, and whether the
workload is exact, approximate, sharded, sliced, or replicated.

## Coding Bias

- Prefer reusable runtime, planner, and IR features over benchmark-specific
  shortcuts.
- Prefer fail-closed behavior over optimistic claims.
- Prefer precise blockers over vague TODOs.
- Prefer one coherent FlagQuantum abstraction over parallel project silos.
- Preserve CPU and single-GPU usability even while developing distributed code.

## Current Strategic Priority

The highest-value work is to close real training loops for:

1. sharded statevector forward and backward
2. sharded MPS site/bond forward and backward
3. sliced or partitioned TN contraction and reverse-mode parameter gradients
4. PyTorch-facing JAX kernels with clear DLPack/autograd boundaries
5. deployment of trained parameterized IR to quantum cloud providers

Do not spend time on distributed-looking work that cannot help a single
too-large quantum AI workload fit, train, or deploy.

## Execution Environment Safety

- All external commands must be bounded by an explicit timeout when hanging is plausible.
- Do not retry the same failed operation more than twice.
- Do not use privilege escalation for ordinary repository operations.
- On timeout, terminate stale child processes before continuing.
- Check repository locks, test-runner locks, temporary files, and sandbox permissions.
- Long-running distributed commands must emit phase heartbeat, last operation,
  memory trend, collective state, and rank activity. On genuine no-progress,
  classify the cause, retain diagnostics, terminate stale participants, and
  verify cleanup. Bounded compile/checkpoint phases use explicit larger budgets.
- Application restart is a last resort, not the default recovery action.
- A session must never claim completion when repository writes or required tests were unavailable.
