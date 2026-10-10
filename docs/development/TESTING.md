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

Those budgets are the claim, and the harness that runs a fault script under
`torchrun` has a deadline of its own. That deadline must outlast the budget it
wraps: below it, the harness reports `subprocess.TimeoutExpired`, which names
neither the operation that stalled nor the launcher that failed to act, and on a
shared runner that is what a busy host produces. `--mode timeout` is the only
mode that asks for a short budget; every other mode must not inherit one.
`tests/unit/test_runtime_harness_deadlines.py` pins the ordering, and
`tests/distributed/statevector_training_runtime.py` lists each mode's budget.

A deadline cannot be made tight enough to assert anything on a shared machine,
so it is not asked to. Two ranks starting, importing torch, and building a
process group measured about 8s idle and about 38s with the host oversubscribed
five to one; the allowance is sized from that measurement, and a hang, being
unbounded, is still caught.

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
| Multi-node transport testing | `python tools/ci_tier.py multinode-scheduled` | Three workloads (statevector, MPS, tensor network), each partitioned across two nodes on one revision, driven by `tools/multinode_launch_plan.py`. | Release claims unless produced payloads also pass release validation. |
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
torchrun --standalone --nproc-per-node=2 -m pytest -m "distributed_launch" -q
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
| `distributed_launch` | One host with a launcher: `torchrun --standalone --nproc-per-node=N`. | Rank-placement and collective semantics for a distributed mode, reachable on CPU. | Real multi-GPU or multi-node capacity. A bare `pytest` selects these and executes none of them. |
| `distributed_accel` | Explicit accelerator or multi-GPU environment. | Hardware-backed distributed behavior for the covered path. | Multi-node transport unless also marked `distributed_multinode`; release claim unless audit evidence passes. |
| `distributed_multinode` | Multi-node cluster, or a scheduled/manual job with explicit rank placement. | Node/rank placement and multi-node transport behavior for the covered path. | Release-grade claim unless benchmark payload and release gate pass. |
| `benchmark_contract` | Fast local pytest over benchmark JSON and audit helpers. | Payload shape, required fields, result hygiene, scanner contract, and non-release payload rejection. | Real benchmark performance or hardware behavior. |
| `release_gate` | Fast local pytest over release validators and promoted payloads. | Fail-closed release validation behavior and accepted release payload contract. | That unmeasured paths are scalable; it only validates provided evidence. |
| `scalability` | Release-grade evidence path, usually scheduled or manual. | A specific promoted payload may support a capacity-scaling claim when it passes the release gate. | Generic distributed readiness for all backends or future workloads. |
| `gpu` | CUDA, vendor, or accelerator device. | Device-specific execution for the covered test. A test that skips on `torch.cuda.is_available()` needs this marker to reach a runner that has a device; without it the CPU lanes select and skip it and the accelerator lane never sees it. | Distributed semantics unless paired with distributed markers. |
| `distributed` | Umbrella marker for distributed planning, runtime, or evidence tests. | The test touches distributed behavior in some form. | Which environment is required; use `distributed_cpu`, `distributed_accel`, or `distributed_multinode` for CI policy. |
| `jax` | The `jax` extra installed (`.[dev,jax]`); selected by the `jax-optional` job and by the coverage job's marker expression. | JAX kernel execution, the hybrid JAX/PyTorch layer, and the sharded MPS, statevector, and tensor-network plans and executors. | That JAX ships in the core distribution; the core lanes prove it is absent. |
| `triton` | The `cuda` extra installed (`.[dev,cuda]`); selected by the `triton-optional` job, which has no device, and by the accelerator tier. | The Triton kernel launch wrappers and the CPU fallbacks beside them. Tests that launch a kernel also carry `gpu`. | That a device is present; a CUDA build is not a GPU. |
| `qiskit` | The `qiskit` extra installed; selected by the `qiskit-optional` job on the certified 2.0.x and 2.5.x lanes. The coverage job excludes it explicitly (`and not qiskit`). | The machine-readable interoperability contract plus real Qiskit IR, statevector, wire-order, classical-bit, fixed-seed bidirectional differential programs, and local Aer conformance. | Hardware submission, or that Qiskit is a core dependency. |
| `pennylane` | The `pennylane` extra installed; selected by the `pennylane-optional` job on the 0.44.1 and 0.45.1 lanes, and by the coverage job, which installs the extra. | The IR-only contract, golden and fixed-seed differential complex128 QuantumScript semantics, and explicit local `lightning.qubit` statevector/shot execution. | Hardware submission, dynamic execution, differentiation, automatic routing, or that PennyLane is a core dependency. |
| `cirq` | The `cirq` extra installed; selected by the `cirq-optional` job on the 1.6.1 and 1.7.0 lanes, and by the coverage job. | Static `cirq.Circuit` conversion, explicit qubit order, fail-closed diagnostics, seeded bidirectional statevector conformance, and explicit local Simulator statevector/shot execution. | Cirq devices, automatic backend routing, or that Cirq is a core dependency. |
| `cudaq` | The isolated `cudaq` heterogeneous-toolchain extra installed; selected by the `cudaq-optional` Linux job on the 0.15.1 and 0.16.0.post1 lanes. | One-way static kernel construction, fail-closed diagnostics, explicit bit-order conversion, and seeded statevector conformance. | Reverse import, runtime execution, targets, GPU/QPU support, or that CUDA-Q is a core dependency. |
| `braket` | The `braket` extra installed for static-circuit tests; selected by the `braket-optional` job on the 1.117.0 and 1.127.1 lanes, and by coverage. Provider tests still use fakes. | Static `braket.circuits.Circuit` conversion, explicit qubit and statevector order, fail-closed diagnostics, seeded bidirectional conformance, and fake-based provider orchestration. | AWS task submission, credentials, QPU behavior, or that Braket is a core dependency. |
| `azure` | The `azure` extra installed (`.[dev,azure]`); selected by the `azure-optional` job on the certified QDK 1.32.3 lane, outside the coverage expression. | The fake workspace and target provider path, the optional `qdk[azure]` import surface and its fail-closed diagnostics, and that the core import loads no part of the Azure stack. | An Azure credential or workspace, network submission, QPU behavior, or that QDK is a core dependency. |
| `slow` | Any environment, intentionally slower than default loops. | Longer-running behavior selected explicitly. | Release readiness or scalability on its own. |

The coverage job installs `jax`, `braket`, `cirq`, `pennylane`, `cotengra`,
`stim`, and `pymatching` because its marker
expression selects their suites, and installs neither `qiskit` nor `triton`: the
other two cannot be measured there.
`cotengra` is a pure-Python wheel whose only dependency is `autoray`, so it does
not add to the static-TLS budget that rules Qiskit out. `stim` carries no marker
of its own and its tests are marked `unit` and `integration`, so the expression
above already selects them; the extra appears on the install line because
otherwise every one of those tests skips at import and `simulation.stabilizer`
measures as uncovered rather than as untested.
`pymatching` is named in the list for the same reason and carries no marker
either: the adapter tests are `integration`, they skip at import without the
extra, and the module they exercise would then measure as uncovered rather than
as untested. The install line is what
`tests/unit/test_lane_dependency_policy.py` reads when it asserts that a lane
which selects a test installs what the test probes.
Qiskit's native
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
- evidence revision provenance: the revisions the walked evidence artifacts
  record, against the declaration in `evidence-revision-origins.toml`;
- multi-team ownership: the policy, and the classification of every path the
  branch changes relative to `origin/main`;
- `pr-default`, `pr-runtime`, and `pr-distributed`.

Run the same gate directly when diagnosing a failure:

```bash
python tools/pre_push.py
python tools/pre_push.py --dry-run
```

The gate stops at the first failure. It is CPU-safe and does not claim to
replace the remote Python-version matrix, clean distribution installs,
supply-chain checks, or accelerator jobs.

The lazy-import step budgets the *fastest* of its samples, not the mean or the
median. Every sample passes through CPython interpreter startup, which measured
88% of the wall clock on an idle host and moves with machine load rather than
with anything this repository does; the floor is the part of the distribution
that stays attributable to the code. A dependency-policy leak exits `2` with the
loaded modules on stderr, so it cannot be read as a slow import.

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

### Opt-In External Source Conformance

The Kaiwu integration is not a declared FlagQuantum dependency while its API
proposal remains unapproved. Its conformance tests therefore run only through
the pinned, credential-free source runner, which validates both Git revisions,
places their `src` directories on `PYTHONPATH`, and sets the following explicit
gate:

| Variable | Tests | What it runs |
| --- | --- | --- |
| `FLAGQUANTUM_TEST_KAIWU_SOURCE` | `tests/team/ecosystem/test_kaiwu_community_conformance.py`, `test_kaiwu_pytorch_plugin_conformance.py`, and `test_qdiffusion_live_system_probe.py` | `examples/qdiffusion_kaiwu/run_local_conformance.sh` against the reviewed Kaiwu Community and Kaiwu PyTorch Plugin source revisions; it does not install or invoke the proprietary SDK |

An ordinary test lane must not set this variable or resolve an arbitrary public
package named `kaiwu`. If the integration is approved as an installable extra,
replace this source-only gate with the reviewed dependency and CI lane rather
than keeping two paths.

### Multi-Node Work

Use this tier only on the launch host, which is the node that can reach its
peer over ssh; the pair holds a device on each of two shared hosts.

Two different things are called multi-node work here, and only one needs a
second node:

- **Launched single-host tests.** `tests/distributed/test_runtime_modes.py`,
  `test_hybrid_jax_runtime.py`, and `test_statevector_correctness.py` carry
  `distributed_launch`. They skip unless a launcher starts them, and they need
  one host, not two. The `cpu-core` job runs them with
  `torchrun --standalone --nproc-per-node=2 -m pytest -m distributed_launch -q`.
  **A bare `python -m pytest -m distributed_launch -q` selects them and executes
  none of them**, which is what this section asked for until 2026-09-18, and
  which let ten tests report skips while every lane stayed green.
  That lane is the PyTorch-only environment, which does not install NumPy, so a
  launched test may not use a Python object collective: `all_gather_object` and
  its siblings move their payload through `Tensor.numpy()`, and the first real
  execution of this lane failed on exactly that. `tests/distributed/conftest.py`
  turns the rule into a failure instead of a surprise, and
  `flagquantum.runtime.executors.mps.metadata_transport.all_gather_json` is the
  tensor-native gather to reach for.
- **Real two-node transport.** One rank per node, the same rendezvous address,
  and `tools/probe_cuda_multinode_statevector.py` on both.
  The probe retains one statevector shard per rank during execution, exports the
  whole amplitude vector as the product of a leg of its own rather than gathering
  it to check an answer, and records the selected NCCL route from a rank-zero
  debug log. It covers the whole training path on that
  shard: forward, the adjoint gradient, an owner-sharded optimizer step, and a
  checkpoint that a restarted run resumes from. Its output is correctness and
  communication evidence only; it is not a scalability or release claim.
- **Real two-node MPS transport.** The same pair, the same launcher, and
  `tools/probe_cuda_multinode_mps.py`: one logical six-wire matrix product state
  split by site, three owned sites per rank, with the middle adjacent gate
  straddling the ownership boundary. It covers the same training path against an
  exact complex128 statevector reference, and its reverse runs the compiled site
  buckets so the layer-boundary halo is prefetched across the host boundary
  rather than reconstructed locally. The training legs run eagerly by design,
  because a checkpointed step is not a compiled-kernel step.
- **Real two-node tensor-network transport.** The same pair and launcher again,
  with `tools/probe_cuda_multinode_tn.py`: one fourteen-wire contraction bound to
  the release contract's own narrowest matched-speed rung, cut on the two labels
  its entangling gate contracts, four slices, two owned per rank. The
  cut is declared rather than chosen by the memory-driven automatic slicer,
  because the cheapest cut on this circuit is carried by a state-copy node whose
  value is zero on every branch but one -- a partition that would report
  `sharded_across_ranks` while a single rank did the arithmetic. The probe
  records four amplitudes, one single-amplitude projection through the other
  output path, the expectation with its gradient, and the training path with
  checkpoint resume, each against the exact complex128 statevector. It also reads
  the committed checkpoint generation back and requires one integrity-checked
  file per rank whose payload agrees with its sidecar.

`tools/multinode_launch_plan.py --run` drives that pair, one workload per
invocation selected by `--probe` (`statevector`, the default, `mps`, or `tn`),
and the `multinode` job in `.github/workflows/scheduled-hardware.yml` is one
call to it per workload.
It runs manual dispatch only, on the runner label that only the launch host
carries, because SSH reaches the peer from there and not the other way round.

It stages the tree onto a filesystem both nodes mount, so the two ranks cannot
disagree about which revision they are evidence about, and it runs them through
`tools/run_multinode_watchdog.py`, so a launcher that dies on one node does not
leave the other waiting for a rendezvous that never arrives. Before any of that
it refuses to launch on nine questions that a timeout would otherwise answer,
among them whether this host is the launch host, whether the peer has the
device and the interface, and whether the rendezvous port is free. The
rendezvous port is chosen rather than assumed: the hosts are shared, and a
preflight against the fixed 29500 refused a real launch because a neighbour
already held it.

The checkpoint directory is a third shared path, and the lane empties it along
with the output directory before the ranks start. A checkpoint left by an
earlier run would be resumed from rather than overwritten, and a stale rank
record in the output directory would be published as this run's evidence if this
run failed before writing its own. Both are named `fq-multinode*`, because the
lane removes them and only a directory with that name is treated as its own.

```bash
python tools/ci_tier.py multinode-scheduled
# what that tier runs, on the launch host:
python tools/multinode_launch_plan.py --run --staging /nfs/fq-multinode-tier --checkpoint-directory /nfs/fq-multinode-tier-checkpoints --report-directory hardware-run --interface ens22f0 --transport rdma --measure
python tools/multinode_launch_plan.py --run --probe mps --staging /nfs/fq-multinode-tier --checkpoint-directory /nfs/fq-multinode-tier-checkpoints --report-directory hardware-run-mps --interface ens22f0 --transport rdma --measure
python tools/multinode_launch_plan.py --run --probe tn --staging /nfs/fq-multinode-tier --checkpoint-directory /nfs/fq-multinode-tier-checkpoints --report-directory hardware-run-tn --interface ens22f0 --transport rdma --measure
```

Each command names its route and asks for the measured leg. `--transport rdma`
configures both ranks for the fabric and disables the socket transport rather
than leaving the choice to NCCL: a lane that let the transport be chosen could
record a socket run and leave the fabric untested while still reporting a pass.
The launch keeps the NCCL debug log either way, and the probe reads the route
back from it, so a log that does not confirm the configured route fails the
probe instead of being recorded as the route the plan asked for. `--measure`
adds a warmup and repeated samples between device synchronizations, so the
artifact carries a measured duration rather than one unsynchronized reading.

The tier passes no `--source-revision`, because the launch host stages from a
git checkout and the probe reads the revision from it. A host that stages from a
copied tree instead -- one without `.git`, which is what a rebuild host usually
has -- must export `FLAGQUANTUM_SOURCE_REVISION` for the tier, or every probe
fails closed on a revision it cannot resolve.

No test carries `distributed_multinode`. The seven that did need a launcher
rather than a second node and were renamed to `distributed_launch`; the marker
stays declared because the two-node evidence comes from the probe above rather
than from pytest. This tier used to run `pytest -m distributed_multinode`, which
selected nothing and reported success having executed nothing.

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
The checked-in `ci.yml` defines twenty jobs:

- `quality`: Ruff and Black over `flagquantum/`, `tests/`, and `tools/`, the
  strict type check of the whole package and of the CI tooling, plus
  dependency-policy synchronization, architecture-boundary, generated-document,
  capability-maturity, evidence-revision-origin, Braket/Cirq/CUDA-Q/Qiskit/
  PennyLane interoperability contracts, and repository-hygiene checks;
- `cpu-core`: Python 3.10 and 3.11 compile, import, dependency-isolation, and
  smoke compatibility on pull requests; pushes to `main` retain the complete
  smoke/unit suite on both interpreters;
- `cpu-core-3-12-tests`: the complete smoke/unit suite on the newest supported
  interpreter;
- `cpu-runtime`: the Python 3.12 integration tier and launched CPU-distributed
  proofs, running in parallel with the complete unit suite;
- `cpu-core-3-12`: the required aggregation gate, reported to branch protection
  as `cpu-core (3.12)`, that succeeds only when both Python 3.12 suites pass;
- `jax-optional`: the JAX extra and its focused hybrid/distributed regression;
- `triton-optional`: the `cuda` extra and the Triton kernels that run without a
  device; the ones that launch a kernel belong to the accelerator tier;
- `cirq-optional`: static circuit conversion, fail-closed diagnostics, seeded
  bidirectional statevector conformance, and explicit local Simulator
  statevector/shot execution against the certified Cirq Core 1.6.1 and 1.7.0
  lanes, isolated from the core environment;
- `braket-optional`: static circuit conversion, fail-closed diagnostics, and
  seeded bidirectional statevector conformance against Amazon Braket SDK
  1.117.0 and 1.127.1, without credentials or cloud submission;
- `azure-optional`: the `azure` extra and the offline Azure provider suite
  against the certified QDK 1.32.x line, with a fake workspace and target only —
  no credential, no workspace resource id, and no submission;
- `cudaq-optional`: one-way kernel export and seeded statevector conformance
  against CUDA-Q 0.15.1 and 0.16.0.post1 on Linux;
- `qiskit-optional`: the machine-readable interoperability contract plus real
  Qiskit IR, statevector, wire-order, classical-bit, fixed-seed bidirectional
  differential programs, and local Aer conformance on the certified Qiskit
  2.0.x and 2.5.x lanes, isolated from the core environment;
- `pennylane-optional`: the IR-only contract, complex128 QuantumScript
  semantics, and explicit local `lightning.qubit` statevector/shot execution
  against the minimum 0.44.1 and latest 0.45.1 supported lanes;
- `dependency-bounds`: the oldest supported Python and Torch line beside the
  newest, so a declared lower bound is exercised rather than assumed;
- `package`: wheel/sdist construction, forbidden-content inspection, and a
  commit/environment/checksum manifest;
- `release-wheel-platform`: the exact release-wheel route on macOS arm64 and
  Windows amd64, including native compilation, runtime smoke tests, repaired
  wheel inspection, and uploadable-artifact verification;
- `distributed-cpu`: the `pr-distributed` tier — CPU distributed semantics and
  release-contract checks — run separately from the core matrix;
- `coverage-shard`: eight deterministic, duration-balanced partitions of the
  maintained package measured under the same marker expression on separate
  runners;
- `coverage`: the required aggregation gate, which combines all raw coverage
  data files before enforcing the floors in `contracts/coverage-policy.toml`
  with `tools/check_coverage.py`;
- `supply-chain`: `pip-audit`, `bandit`, and a validated CycloneDX SBOM.

Each two-version ecosystem compatibility job reuses one runner for its minimum
and latest SDK checks. The versions still execute in separate Python and pytest
processes, but checkout, Python setup, the PyTorch wheel, and the editable
FlagQuantum install happen once per framework instead of once per version. This
preserves the certified compatibility matrix while reducing runner cold starts
and the serial queue behind the `light-b` and `light-c` capacity buckets.

The GitHub-hosted jobs are partitioned across sixteen concurrency buckets per
pull request or pushed branch. The scope is part of every bucket name: it keeps
one run at sixteen concurrent jobs without forcing a new pull request to wait for
an older pull request's coverage or interpreter lane. The organisation-level
20-runner pool remains the aggregate admission limit. Superseding a run of the
same branch still cancels the older run.

Optional integration jobs name the test files they own and retain the marker
selector inside that file set. This prevents an unrelated module-level
collection error elsewhere in the repository from making every Cirq, CUDA-Q,
PennyLane, Qiskit, and Triton lane fail before its own tests start. The broad
CPU and coverage lanes remain responsible for repository-wide collection.

The pull-request `pre-commit` workflow runs only source-hygiene hooks. Ruff,
Black, mypy, architecture, generated-contract, and repository-wide policy
checks are owned by the `quality` job and are not repeated in a second full
project environment. The local pre-commit configuration remains broader so a
developer still gets those checks before pushing.

Each CPU core and coverage-shard pytest phase uses exactly two xdist workers
with work-stealing scheduling. The fixed count shortens the critical path
without letting a runner-image CPU change silently widen process fan-out. The
eight coverage shards use checked-in duration hints from a cited CI run and a
deterministic least-loaded assignment, then merge raw data before applying the
unchanged coverage policy; their union is the former selection, not a reduced
test set. Unmeasured tests receive a small default weight, so new tests remain
evenly distributed without making the historical timing data authoritative.
Pull requests run complete smoke/unit coverage once on
Python 3.12 while Python 3.10 and 3.11 run the compatibility smoke; pushes to
`main` retain complete smoke/unit coverage on all supported interpreters. The
integration tier runs once on Python 3.12, in parallel with its unit suite,
because it proves repository runtime behavior rather than interpreter
compatibility.
Only the ordinary smoke, unit, integration, and coverage phases are
parallelized: `torchrun` and `distributed_launch` remain explicit serial
workflow steps on Python 3.12 so xdist never creates a second process topology
inside a launched rank. Both parallel phases report their fifty slowest tests
to keep the next optimization grounded in measured durations.

The full P5 optimizer-trajectory and autograd conformance tests carry both
`integration` and `slow`. On pull requests they run once in the Python 3.12
`cpu-runtime` integration tier; the instrumented coverage phase excludes `slow`
to avoid repeating the same 96-119 second proofs. A push to `main` runs the
complete coverage selector, including those tests, so the post-merge coverage
artifact and package floors still measure their production paths.

The `cudaq` extra has its own Linux SDK matrix because its platform-specific
toolchain is intentionally absent from portable and coverage environments.
The lane is the evidence for the two declared tested versions; macOS is not a
supported package target in this boundary.

### `--standalone` does not rendezvous on macOS

Every test that spawns ranks with `torch.distributed.run --standalone` fails on
macOS with a subprocess timeout, and the cause is outside this repository. Two
lines reproduce it:

```python
# /tmp/gloo_probe.py
import torch.distributed as dist

dist.init_process_group("gloo")
print(dist.get_rank(), dist.get_world_size(), flush=True)
dist.destroy_process_group()
```

```console
$ python -m torch.distributed.run --standalone --nproc-per-node=2 /tmp/gloo_probe.py
# hangs; killed after 60 s
[W socket.cpp:764] [c10d] The IPv6 network addresses of
(1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa, 55738)
cannot be retrieved (gai error: 8 - nodename nor servname provided, or not known).
```

`--standalone` picks the wildcard address, the reverse lookup of its IPv6 form
fails, and c10d retries with exponential backoff rather than falling back. On
this Mac the consequence is that `python tools/ci_tier.py release`, whose first
command is `pytest -m "not scalability"`, cannot pass: it reports 14 failures,
all of them `subprocess.TimeoutExpired` in
`tests/distributed/test_mps_stability_faults.py`. Those tests, their runtime
helper, and the MPS executor they exercise are untouched by the campaign that
observed this, and the probe above imports nothing from FlagQuantum, so the
failures carry no information about the repository. Run the `release` tier on
Linux, or select the suite's markers directly and read the macOS result as
partial.

The Braket circuit adapter owns only static object conversion. Its isolated SDK
matrix proves both declared versions and local unitary conformance. Existing
fake-based provider tests separately prove local submission orchestration and
result normalization; neither suite submits an AWS task or validates hardware.

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
| Multi-node scheduled/manual | Manual dispatch on the launch host, which is the node that can reach its peer over ssh. | `python tools/ci_tier.py multinode-scheduled` | No | One workload partitioned across two nodes on one revision; release claims still require audit payloads. |
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
self-hosted label, together with `local-scale-scheduled` and
`crossover-scheduled` in `scheduled-hardware.yml`. Two runners carry that label
as of 2026-09-18, one on `jp-a800-171` and one on `jp-a800-172`; confirm with
`gh api repos/flagos-ai/FlagQuantum/actions/runners`, because a statement about
what is registered is only true of the moment it was written.

Both hosts are shared with other people's workloads, and a runner label says
`gpu` rather than how many devices its host has. Every job that requests a
device therefore joins one `flagquantum-accelerator` concurrency group with
`queue: max`, so that no two accelerator jobs measure the same host at once and
the four- and eight-device matrix legs take turns instead of both landing on one
host. A lane that waits stays queued: the default keeps one waiting lane and
drops it when another arrives, which reports nothing at all for the one dropped.

The clone those jobs check out is shared for the same reason the hosts are: the
runner keeps one working directory per repository and reuses it for every job,
so what one job does to `.git` outlives it. `actions/checkout@v4` cannot undo a
sparse checkout on the git these runners carry (2.34). It runs `git
sparse-checkout disable`, which restores the files and clears the skip-worktree
bits, but leaves `core.sparseCheckout` set and the pattern file in
`.git/info/sparse-checkout`; the `git checkout --force -B <branch> <sha>` that
follows in the same step then re-applies those patterns, and `git clean -ffdx`
and `git reset --hard` do not repair the tree because both honour the
skip-worktree bits. A job that does not name its own paths can therefore find
the tree pruned by whatever ran before it, and see a successful checkout. That
is how `local-scale-scheduled` got six phases through on 2026-09-18 and then
could not open `benchmarks/mps_boundary_transport.py`, a file `local-gpu.yml`'s
pattern list does not name. Every self-hosted job here that does not pass
`sparse-checkout:` clears the leftover state before it checks out, and
`tests/unit/test_shared_runner_checkout_policy.py` holds both halves of that
rule.

**A pull request must not be able to reach a self-hosted runner.** The runners
are root shells on hosts that other people share, and one of them serves a
public repository, which is the combination GitHub's own guidance warns about.
No workflow a pull request can trigger asks for a self-hosted runner today —
`ci.yml`, `cd.yml`, and `publish-dev-container.yml` all run on GitHub-hosted
runners, and the three that need a device are `workflow_dispatch` or `schedule`.

That was a property of how the files happened to be written rather than anything
the repository enforced, and there is no runner group to enforce it: a runner
group is an organization-level feature, while both runners here are registered
at the repository level. `tests/unit/test_hardware_lane_policy.py` therefore
holds the rule directly, by failing when a `pull_request`, `pull_request_target`,
or `workflow_run` triggered workflow names a self-hosted runner. `workflow_run`
counts because it fires in response to a workflow a pull request did trigger, so
it reaches whatever runner it asks for.

Serializing our own lanes does not make a host private. A benchmark records the
device state it found before its first allocation, and
`tools/evaluate_performance_artifact.py --write` reads that record into a
`measurement_validity` block:

- `clean` — every device the run used held at most `foreign_memory_floor_mib`
  MiB, which is consistent with an idle device.
- `contended` — another process held more than that, on a device this run used.
- `unavailable` — the state was never read, or could not be interpreted. This
  is deliberately not `clean`: a run that did not look has not shown that its
  devices were free.

The verdict is recorded rather than folded into the performance gate. Contention
inflates latency variance, and the gate reports that as
`latency_variance_exceeds_threshold`, the same error a real regression produces.
Letting contention turn the gate red would make a busy neighbour look like a code
regression, so the gate keeps judging the numbers while this block says whether
the numbers were taken somewhere that allows believing them. A `contended`
verdict does not by itself invalidate correctness evidence, which contention
does not touch.

Four- and eight-GPU jobs live in `scheduled-hardware.yml`. They are scheduled or
manually dispatched and do not block unrelated documentation or CPU-only
changes. The scheduled entry point is `cron: "17 3 * * 1"`, which is 03:17 UTC
on Monday. Hardware manifests and raw logs are uploaded even on failure; skips
must carry a reason and are not evidence of success, and neither is an artifact
whose `measurement_validity` is not `clean` evidence of performance.

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
