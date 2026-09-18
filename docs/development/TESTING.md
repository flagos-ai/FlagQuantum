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
passes `flagquantum.runtime.audit.release_policy.require_distributed_scalability(...)`
and the benchmark audit commands
for `benchmarks/results/scalability/`.

## Quick Start

| Situation | Command | What It Proves | What It Does Not Prove |
| --- | --- | --- | --- |
| Daily development or any issue baseline | `python tools/ci_tier.py pr-default` | Fast smoke/unit health for imports, minimal circuits, autograd, pure planner/audit helpers. | Runtime integration, distributed behavior, performance, or release readiness. |
| Local API/runtime/planner/compiler changes | `python tools/ci_tier.py pr-runtime` | Seeded `integration` coverage for local runtime/API behavior. | Multi-process transport, GPU execution, or scalability claims. |
| Distributed planner/audit/benchmark contract changes | `python tools/ci_tier.py pr-distributed` | CPU distributed semantics, fail-closed gates, benchmark JSON contracts, release-gate validation. | Real multi-GPU or multi-node capacity expansion. |
| Real multi-GPU or accelerator testing | `python tools/ci_tier.py gpu-scheduled` | Accelerator-backed tests selected by `distributed_accel and gpu`, plus the device-bound Triton kernels selected by `triton and gpu`. | Multi-node transport or release scalability by itself. |
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
python -m pytest -m "triton and gpu" -q
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

Every test function carries a marker some lane selects, so a marker-selection
command selects the whole set it names; `tests/unit/test_test_reachability_policy.py`
walks each file's syntax tree and fails on a test no lane can reach. An earlier
file-level guard was too weak — one marked test rescued fifty unmarked ones —
so the check is per test rather than per file.

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
| `gpu` | CUDA, vendor, or accelerator device. | Device-specific execution for the covered test. A test that skips on `torch.cuda.is_available()` needs this marker to reach a runner that has a device; without it the CPU lanes select and skip it and the accelerator lane never sees it. | Distributed semantics unless paired with distributed markers. |
| `distributed` | Umbrella marker for distributed planning, runtime, or evidence tests. | The test touches distributed behavior in some form. | Which environment is required; use `distributed_cpu`, `distributed_accel`, or `distributed_multinode` for CI policy. |
| `jax` | The `jax` extra installed (`.[dev,jax]`); selected by the `jax-optional` job and by the coverage job's marker expression. | JAX kernel execution, the hybrid JAX/PyTorch layer, and the sharded MPS, statevector, and tensor-network plans and executors. | That JAX ships in the core distribution; the core lanes prove it is absent. |
| `triton` | The `cuda` extra installed (`.[dev,cuda]`); selected by the `triton-optional` job, which has no device, and by the accelerator tier. | The Triton kernel launch wrappers and the CPU fallbacks beside them. Tests that launch a kernel also carry `gpu`. | That a device is present; a CUDA build is not a GPU. |
| `qiskit` | The `qiskit` extra installed; selected by the `qiskit-optional` job on the certified 2.0.x and 2.5.x lanes. The coverage job excludes it explicitly (`and not qiskit`). | The machine-readable interoperability contract plus real Qiskit IR, statevector, wire-order, classical-bit, and local Aer conformance. | Hardware submission, or that Qiskit is a core dependency. |
| `pennylane` | The `pennylane` extra installed; selected by the `pennylane-optional` job on the 0.44.1 and 0.45.1 lanes, and by the coverage job, which installs the extra. | The IR-only contract and complex128 QuantumScript semantics. | Hardware submission, or that PennyLane is a core dependency. |
| `braket` | No extra required: the provider surface is exercised against fakes, and the nightly tier selects these tests. | The Amazon Braket provider and dynamic-deployment surface. | Hardware submission, or any real SDK or device behavior. |
| `slow` | Any environment, intentionally slower than default loops. | Longer-running behavior selected explicitly. | Release readiness or scalability on its own. |

The coverage job installs `jax`, `pennylane`, and `cotengra` because its marker
expression selects their suites, and installs neither `qiskit` nor `triton`: the
braket tests need no extra, and the other two cannot be measured there.
`cotengra` is a pure-Python wheel whose only dependency is `autoray`, so it does
not add to the static-TLS budget that rules Qiskit out. Qiskit's native
libraries cannot be loaded in that process at all — `qiskit/_accelerate.abi3.so`
raises `ImportError: cannot allocate memory in static TLS block` once the rest of
the test tree has been imported, and importing it first only moves the failure to
`qiskit_aer.libs/libgomp-*.so`. pytest reports that as a collection error, which
aborts the lane outright. The Triton kernels are omitted from measurement in
`.coveragerc`, so installing the extra there would buy no coverage. The
expression excludes both markers, and the `qiskit-optional` and
`triton-optional` jobs remain the lanes that run those tests.

Excluding a marker reaches the tests that probe the package inside the test
body. Files that gate the whole module on `importorskip` call it during import,
before the marker filter, so each still records one module-level skip naming its
cause. A self-describing skip costs nothing. The failure worth guarding against
is a suite that skips everywhere and leaves its package measuring far below what
it covers, which is what `pennylane` was doing at 43.4%.

Because a missing extra is swallowed as a skip, the two halves of every job —
the install line and the marker expression — are checked against each other by
`tests/unit/test_lane_dependency_policy.py`. It holds five invariants. The
coverage job must install what it selects, or its measurement is a lie. Every
optional integration some test probes must be installed by *some* lane that
selects that test. And a test that waits on a device must be within reach of a
lane that has one — the six Triton files sat in no lane at all while carrying
`unit`, and six more tests in `tests/unit/test_real_imag_kernels.py` carried a
CUDA guard without the `gpu` marker, so the CPU lanes selected and skipped them
while the accelerator lane never selected them at all. The fourth closes the
hole the second leaves: a probe for a package the policy declares nowhere is not
"no lane installs this" but "no lane can be wired to install it", so the answer
has to be recorded in `dependency-policy.toml` rather than implied by silence.
The fifth applies the same reasoning to the variable that gates an opt-in device
run: no lane sets one, so the table under "Opt-In Device Runs" is where it has to
be recorded.
Add an extra to an install line, or a marker to a selector; do not let the skip
absorb the difference.

What counts as an optional integration comes from `dependency-policy.toml`, so
the check cannot drift from the policy. A probe for a package the policy declares
nowhere is not outside it — that is the case the fourth invariant reports,
because no lane can be wired to install what nothing declares.

## Test Tiers

### Push Gate

Install both repository hooks once per clone:

```bash
pre-commit install
```

The checked-in configuration installs `pre-commit` and `pre-push`. Before a
push leaves the machine, the push gate runs:

- all normal pre-commit quality and source-of-truth checks;
- the strict type check of the whole package, and of the CI tooling;
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
python -m pytest -m "triton and gpu" -q
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

### Opt-In Device Runs

These tests are skipped unless an environment variable is set, and no lane sets
one. That is deliberate for a hardware route a CI runner cannot represent, but
it means the variable is the only way in: without it the test skips in every
lane, and a skip reads the same as a pass. Set the variable on a host that has
the device and run the file directly.

| Variable | Test | What it runs |
| --- | --- | --- |
| `FLAGQUANTUM_DOMESTIC_ATTESTATION` | `tests/test_domestic_single_card_certification.py` | `tools/validate_domestic_single_card.py --attestation <value>` over the P0-P5 phases |
| `FLAGQUANTUM_TEST_FLAGOS_CUDA` | `tests/test_flagos_cuda_reference.py` | `tools/validate_flagos_cuda_reference.py` |
| `FLAGQUANTUM_TEST_FLAGOS_DISTRIBUTED` | `tests/test_flagos_distributed_conformance.py` | `tools/validate_flagos_distributed_conformance.py` |
| `FLAGQUANTUM_TEST_FLAGOS_TRANSPORT` | `tests/test_flagos_transport_observability.py` | `tools/observe_flagos_transport.py` |
| `FLAGQUANTUM_TEST_SPLIT_DEVICE_DOUBLE_SINGLE_FLAGOS` | `tests/test_split_real_imag_device_double_single_flagos.py` | `tools/validate_split_real_imag_device_double_single_flagos.py` |
| `FLAGQUANTUM_TEST_SPLIT_DOUBLE_SINGLE_CUDA` | `tests/test_split_real_imag_double_single_conformance.py` | the P3 double-single conformance on `cuda:0`; its CPU sibling needs no variable |
| `FLAGQUANTUM_TEST_SPLIT_DOUBLE_SINGLE_FLAGOS` | `tests/test_split_real_imag_double_single_flagos.py` | `tools/validate_split_real_imag_double_single_flagos.py` |
| `FLAGQUANTUM_TEST_SPLIT_FLAGOS` | `tests/test_split_real_imag_flagos.py` | `tools/validate_split_real_imag_flagos.py` |
| `FLAGQUANTUM_TEST_P5_OPTIMIZER_DEVICE` | `tests/test_split_real_imag_optimizer_accelerator.py` | `tools/validate_split_real_imag_optimizer_accelerator.py --device <value>`, where the value is `cuda:0` or `flagos:0` |
| `FLAGQUANTUM_TEST_SPLIT_PRECISION_CUDA` | `tests/test_split_real_imag_precision_conformance.py` | the P2 precision conformance on `cuda:0`; its CPU sibling needs no variable |
| `FLAGQUANTUM_TEST_SPLIT_PRECISION_FLAGOS` | `tests/test_split_real_imag_precision_flagos.py` | `tools/validate_split_real_imag_precision_flagos.py` |
| `FLAGQUANTUM_TEST_SPLIT_CUDA` | `tests/test_split_real_imag_training_conformance.py` | the P1 training conformance on `cuda:0`; its CPU sibling needs no variable |
| `FLAGQUANTUM_TEST_SPLIT_TRAINING_FLAGOS` | `tests/test_split_real_imag_training_flagos.py` | `tools/validate_split_real_imag_training_flagos.py` |

Most of these set the variable to `1`; the two that take a value say so above.
Each entry asserts the JSON the tool prints, including its fail-closed claims,
so the wrapper is where a tool that started overclaiming gets caught.

`tests/unit/test_lane_dependency_policy.py` fails if a test gates itself on a
variable this table does not name, so a new opt-in run shows up here rather than
skipping quietly.

### Multi-Node Work

Use this tier only in a scheduled/manual torchrun or cluster environment with
explicit rank placement.

For a minimal two-node CUDA correctness check, launch one rank per node with
the same rendezvous address and run `tools/probe_cuda_multinode_statevector.py`.
The probe retains one statevector shard per rank during execution, materializes
the tiny five-wire state only for validation, and records the selected NCCL
route from a rank-zero debug log. Its output is correctness and communication
evidence only; it is not a scalability or release claim.

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
The checked-in `ci.yml` defines eleven jobs:

- `quality`: Ruff and Black over `flagquantum/`, `tests/`, and `tools/`, the
  strict type check of the whole package and of the CI tooling, plus
  dependency-policy synchronization, architecture-boundary, generated-document,
  capability-maturity, and repository-hygiene checks;
- `cpu-core`: Python 3.10-3.12 smoke/unit and integration with core dependencies
  only, including proof that importing and differentiating a native circuit
  does not import JAX;
- `jax-optional`: the JAX extra and its focused hybrid/distributed regression;
- `triton-optional`: the `cuda` extra and the Triton kernels that run without a
  device; the ones that launch a kernel belong to the accelerator tier;
- `qiskit-optional`: the machine-readable interoperability contract plus real
  Qiskit IR, statevector, wire-order, classical-bit, and local Aer conformance
  on the certified Qiskit 2.0.x and 2.5.x lanes, isolated from the core
  environment;
- `pennylane-optional`: the IR-only contract and complex128 QuantumScript
  semantics against the minimum 0.44.1 and latest 0.45.1 supported lanes;
- `dependency-bounds`: the oldest supported Python and Torch line beside the
  newest, so a declared lower bound is exercised rather than assumed;
- `package`: wheel/sdist construction, forbidden-content inspection, and a
  commit/environment/checksum manifest;
- `distributed-cpu`: the `pr-distributed` tier — CPU distributed semantics and
  release-contract checks — run separately from the core matrix;
- `coverage`: the maintained package measured under the marker expression above,
  with the floors in `contracts/coverage-policy.toml` enforced by
  `tools/check_coverage.py`;
- `supply-chain`: `pip-audit`, `bandit`, and a validated CycloneDX SBOM.

Two further items are not jobs but placement rules for tests that run inside the
CPU tiers: Double-Single primitives run in the ordinary CPU unit/integration
tiers, with the same conformance marked `distributed_accel and gpu` for the
scheduled CUDA runner, without turning that result into vendor certification;
and split real/imag statevector P0 runs in the ordinary CPU unit/integration
tiers, with its CUDA and Torch-FL `flagos:0` tests GPU-marked portability
evidence that does not certify a domestic accelerator or provider-internal
route.

`scheduled-hardware.yml` is the explicit self-hosted GPU and manual multi-node
entrypoint. A skipped hardware test remains wiring information only, never
execution proof.

| CI Tier | Trigger Condition | Command | Blocks Ordinary PRs | Proof Boundary |
| --- | --- | --- | --- | --- |
| PR default | Every pull request and ordinary local issue verification. | `python tools/ci_tier.py pr-default` | Yes | Fast `smoke or unit` health only. |
| PR runtime/API/planner | Pull requests that touch runtime, API, planner, compiler, or integration boundaries. | `python tools/ci_tier.py pr-runtime` | Only when selected for that blast radius. | Seeded `integration` runtime/API files; not distributed or scalability evidence. |
| PR distributed/audit/benchmark | Pull requests that touch distributed planning/runtime metadata, audit, benchmark JSON, or release gates. | `python tools/ci_tier.py pr-distributed` | Only for distributed/audit/benchmark changes. | CPU distributed semantics plus benchmark contract and release-gate validation; not capacity evidence. |
| Nightly CPU | Scheduled nightly CPU-safe validation or explicit maintainer request. | `python tools/ci_tier.py nightly` | No | Broad non-benchmark, non-accelerator, non-multinode health. |
| GPU scheduled | Scheduled/manual job on an explicit accelerator runner. | `python tools/ci_tier.py gpu-scheduled` | No | Accelerator-backed behavior for selected tests, including the Triton kernels that need a device; not multi-node or release evidence by itself. |
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

`AGENTS.md` is the concise agent-facing workflow for this tiered policy. It
should stay consistent with this manual: start from
`python tools/ci_tier.py pr-default`, then expand by blast radius. Focused
file-list commands are still useful for review-sized checks, but tier commands
are the official vocabulary.

## Non-Negotiable Release Boundary

CPU distributed tests prove semantics and fail-closed behavior only. They do not
prove real multi-card capacity expansion. Never use CPU distributed tests alone as scalability release evidence. Release claims require accelerator or
multi-node evidence plus benchmark audit and release payload validation.
CPU distributed tests are not scalability evidence and must not be used as release-grade scalability evidence.
### Local GPU gates

`.github/workflows/local-gpu.yml` is `workflow_dispatch`-only: it has no path
filter and no schedule, so it runs when someone starts it. Its one-GPU behavior
job and two-GPU correctness/autograd job both require the `flagquantum-local`
self-hosted label. As of 2026-09-18 no runner carries that label — the
repository has zero registered self-hosted runners and the organization list is
not readable without `admin:org` — so every run of those two jobs has queued and
been cancelled at GitHub's 24-hour limit rather than executed. The same label
gates `local-scale-scheduled` and `crossover-scheduled` in
`scheduled-hardware.yml`.

That makes the two-GPU job the required hardware gate for distributed-runtime
changes only if the label is provisioned. Until it is, a distributed-runtime
change cannot obtain the evidence this section describes, and the run history
says so; nothing in the repository can substitute CPU tests for it.

Four- and eight-GPU jobs live in `scheduled-hardware.yml`. They are scheduled
or manually dispatched and do not block unrelated documentation or CPU-only
changes. Hardware manifests and raw logs are uploaded even on failure; skips
must carry a reason and are not evidence of success.
## Coverage gate

CI measures the maintained package with the seeded smoke, unit, integration, and
JAX tiers, excluding the `qiskit` and `triton` markers for the reasons given
above. `contracts/coverage-policy.toml` is the authority: it sets a global
floor and a per-directory floor for every package, and `tools/check_coverage.py`
enforces both against the XML report. Floors only move upward; raising one is
part of the change that earns it.

The global floor is 76%. Per-directory floors sit above it where a directory is
well covered, so a directory that regresses is caught even while the aggregate
still passes. The current measurement is reported by the `coverage` job rather
than restated here: it moves whenever a test is added, and a number in prose has
nothing keeping it true. Benchmark tooling and GPU-only Triton kernels are owned
by their dedicated benchmark and hardware gates; backend-independent runtime,
accelerator orchestration, and multi-node control paths remain visible in the
XML report.
