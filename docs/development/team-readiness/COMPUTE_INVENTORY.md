# Compute Capability Inventory

Status date: 2026-09-03

Responsible team: Compute

Branch: `codex/vnext-team-platform-providers`

Directory progress (2026-09-04): after the first CPU vertical workflow passed,
`flagquantum/runtime/platforms` moved entirely to `flagquantum/compute`. Repository
callers migrated and the old package was deleted. This physical move does not
promote any capability conclusions below.

## Conclusion

The repository has a small, explicit platform boundary:
`flagquantum.compute.PlatformRuntime` owns device discovery, activation,
synchronization, memory snapshots, streams/events, RNG, and runtime identity.
Built-in implementations are PyTorch CPU, PyTorch CUDA, and lazily loaded Torch-FL
FlagOS. The sole authoritative extension registration/lifecycle mechanism remains
`flagquantum.ecosystem.extensions.sdk.ExtensionRegistry`. This round added no
Provider registration system or protected contract changes.

Distinguish interfaces, software execution evidence, and real hardware certification:

- CPU is an always-available real local path with unit/integration tests and a
  `production_supported` `local_statevector` capability entry. CPU total memory,
  topology, and collectives are not currently discoverable `PlatformRuntime` fields.
- NVIDIA CUDA is a real execution path. The repository contains A100/A800
  single-/multi-GPU and dual-node A800 records. Each establishes only its recorded
  CUDA/NCCL workload, not every CUDA device, operator, or precision combination.
- FlagOS has a real Torch-FL adapter and logical `flagos` execution evidence, but
  checked-in records use physical NVIDIA A800 devices and CUDA-built Torch-FL.
  They establish FlagOS-on-CUDA adaptation/portability, not domestic-card support.
- No reviewed real domestic-card result is checked in. Existing
  `tests/test_domestic_single_card_certification.py` runs only with externally
  supplied `FLAGQUANTUM_DOMESTIC_ATTESTATION`. Even passing candidates retain
  `hardware_certification=false`, `production_claim_allowed=false`, and
  `scalability_claim_allowed=false`.
- `FLAGQUANTUM_ACCELERATOR` hints, Hygon/FlagOS test fakes, and interface types
  establish discovery/isolation logic only. They neither make devices available
  nor establish hardware support.

## Inventory Method

A real capability has an executable implementation plus tests or checked-in
records matching its claimed scope. An adapter capability can discover, activate,
or call a provider through public runtime boundaries while its physical hardware
or internal route may remain uncertified. Interface/mock evidence means protocols,
environment hints, monkeypatches, or gates awaiting external hosts. Interpret
hardware records only within their device, version, node-count, dtype, workload,
and claim-level scope.

This round's code is `single_device_fast_path` platform lifecycle conformance.
The document also cites existing `sharded_across_ranks`,
`rank_local_replicated_kernel`, and collective records, but creates no new hardware
or scalability evidence.

## Platform Capability Matrix

| Platform/evidence level | Discovery and lifecycle | Memory | Kernels/execution | Native double precision | Double-Single software precision | P2P/collectives | Internode communication/topology | CPU fallback | Evidence conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CPU (real) | Always installed/activated/available; one `cpu` device; synchronization, empty stream context, host events, RNG | `MemorySnapshot` fields exist but all CPU values are unknown | Real local PyTorch statevector/training; Platform contract does not enumerate kernels | Local numerical tests use `float64/complex128`; not per-operator certification | CPU P0–P5 experiments are tested; P5 autograd delivers FP32; explicit Double-Single SGD is not `torch.optim` | No Platform communication methods; Runtime owns Gloo/local distributed behavior | No CPU topology contract; CPU distributed tests prove semantics only | Already CPU; no fallback-to-CPU event | Real local capability without hardware capacity/topology claims |
| PyTorch CUDA (real NVIDIA) | `torch.cuda` discovery, device names/memory, synchronization, streams/events, RNG | Allocated/reserved/free/total available; free/total stay unknown on API failure | Real PyTorch CUDA, Triton, local and sharded paths; Runtime/Simulation gate each kernel | A800 records include `complex128`; only recorded operators/workloads, not universal FP64 certification | A800 P0–P4 and explicit P5 SGD portability evidence; still experimental | Existing A800 records execute NCCL, P2P, all-gather/all-reduce/broadcast; interpret each payload's semantics | Single-node 2/4/8/16-card and dual-node A800 development records; some dual-node records remain `development_smoke` or `requires_runtime_summary` | No platform-level automatic CPU fallback; upper layers disclose fallback | Real NVIDIA CUDA support bounded by capability matrix and exact JSON records |
| Torch-FL FlagOS on CUDA (real adapter, no domestic certification) | Imports `torch_fl` only on explicit request; requires registered `torch.flagos`; discovery, sync, streams/events, RNG, identity | Optional allocated/reserved/free/total APIs; missing values stay unknown | Local/sharded statevector and limited training trajectories executed on A800 CUDA-backed Torch-FL | Single-card and 2/4/8-card `complex128` records prove FlagOS-on-A800, not domestic native FP64 | P0–P4 and explicit P5 SGD portability records; provider internals unaudited | Single-node 2/4/8-card all-gather, all-reduce, broadcast, isend/irecv; complex `reduce_scatter_tensor` unsupported in the full matrix; FlagCX routes/host staging unverified | Single-node FlagOS only; no multinode evidence; topology and inner links unattributed | Missing/unavailable requests fail closed; errors mention available CPU/CUDA alternatives but never substitute silently | `development_evidence`/experimental, not domestic hardware, production, or FlagCX certification |
| Real domestic cards (unverified) | Attestation-driven candidate harness and test mocks only | Unverified | Unverified | Unverified | Designed for P0–P5 verification; no checked-in real-card pass | FlagCX unverified | Single-card candidates; no multi-GPU/multinode evidence | Unverified; candidate payloads must prohibit silent fallback | Evidence gap, not support |
| Other vendors/unknown accelerators (interface/mock) | Environment hints have `available=false`; vendor names do not determine types | None | None | None | Device-generic interfaces do not establish device support | None | None | Never selected by auto | Interface/mock only; no hardware claims |

## Facts by Capability

### Device Discovery and Lifecycle

The fixed built-in table in `compute/registry.py` is the current discovery entry.
CPU/CUDA require no optional dependencies; global discovery does not import
inactive FlagOS providers. `resolve_platform_device()` checks availability and
explicit indices before workload allocation, raising `PlatformUnavailableError`
on failure. The extension SDK separately provides a task-local immutable
`ExtensionRegistry` and start/invoke/close failure isolation. These responsibilities
are distinct; no production bridge or duplicate registry was added.

### Memory

CUDA reads allocator allocated/reserved and device free/total memory. FlagOS calls
only optional memory APIs publicly attached to `torch.flagos`; absent values are
not guessed. CPU returns unknown fields. The contract lacks peak memory,
workspace, NUMA/HBM hierarchy, unified memory, OOM categories, and per-rank
aggregation. Those values currently appear in individual Runtime/benchmark payloads.

### Kernels

`PlatformRuntime` does not enumerate kernels, probe operators, or express
compilation/fallback fields. Operator evidence comes from `compute/flaggems.py`,
`runtime/operator_probes.py`, Simulation PyTorch/Triton implementations, and workload
tests. Discovering a provider does not establish statevector, MPS, TN, gradient,
or arbitrary operator support. Automatic FlagOS selection remains disabled until
workload-level operator and numerical evidence passes.

### Precision

The generic backend registry lists `complex64/complex128` as candidate PyTorch
dtypes, not per-device/kernel guarantees. Native double precision requires concrete
execution records. Double-Single uses four FP32 words or high/low accumulation,
retains FP32 exponent range, and is not generally equivalent to FP64/complex128.
P3 still constructs float64/complex128 gates on CPU before transfer; P4 generates
gates on-device for a bounded angle/gate set. Ordinary P5 autograd loss/gradient
delivery remains FP32.

### Communication and Topology

`PlatformRuntime` has no P2P, collective, process-group, rank-placement, or topology
methods. Runtime owns CUDA/NCCL and FlagOS distributed execution. FlagOS identity
intentionally distinguishes `outer_backend=flagos` from `inner_backend=flagcx`.
Connecting a Torch-FL process group does not establish FlagCX use, absence of host
staging, or physical routes. Existing transport observability records verify
logical `flagos` input/output residency and some collective correctness, but
incomplete CUPTI device activity leaves `host_staging_observed` unknown and
disables communication claims.

### Fallback and Evidence

Platform resolution fails closed. Unavailable FlagOS requests never silently
return CPU/CUDA; users may explicitly choose another available platform. Algorithmic
eager, dense, host-reference, or CPU correctness paths must appear in their own
results/evidence and cannot be hidden by platform discovery status. Environment
hints and device names are diagnostic only and do not promote maturity.

## Real Evidence Sources

| Source | Establishes | Does not establish |
| --- | --- | --- |
| `tests/unit/test_platform_runtime.py` | CPU lifecycle; CUDA discovery logic; FlagOS lazy loading, public memory/stream/event/identity adaptation; no automatic unknown-vendor classification | Real CUDA/FlagOS/domestic execution |
| `tests/team/compute/test_platform_consistency.py` | Three built-in runtimes satisfy current `PlatformRuntime` structure; CPU lifecycle consistency; existing SDK device-extension lifecycle without a second registry | Accelerator, kernel, communication, performance, or hardware support |
| `artifacts/flagos_cuda_reference_a800_20260824.json` | Local `flagos:0` reference on NVIDIA A800 and CUDA-backed Torch-FL | Domestic cards, production performance, absence of host fallback |
| `artifacts/flagos_workload_capability_f4_20260826.json` | Single-node 2/4/8-card complex64/complex128 statevector forward/training development evidence; full collective matrix exposes complex reduce-scatter gap | FlagCX routes, multinode, production/release certification |
| `artifacts/flagos_statevector_capacity_f5_a800_20260827.json` | Same 32-qubit complex128 workload OOMs on single-card/replicated paths and completes on eight-rank sharding; physical NVIDIA A800 | Domestic hardware, backward/optimizer capacity, general scalability, FlagCX, absence of host staging |
| `artifacts/flagos_transport_observability_f6_a800_20260827.json` | Single-node 2/4/8-rank four-collective correctness and logical residency for two complex dtypes | Inner communication implementation, FlagCX, host staging, performance, multinode |
| `artifacts/split_real_imag_optimizer_a800_20260825.json` | Experimental explicit Double-Single SGD trajectory portability on native CUDA and FlagOS-on-CUDA | General FP64 equivalence, `torch.optim`, convergence, domestic cards, production support |
| `tests/test_domestic_single_card_certification.py` | Fail-closed candidate fields and expected P0–P5 stages | An existing checked-in real domestic-card pass |
| `capability-maturity.toml` and generated Known Limitations | Claim ceilings and evidence references | Inferences beyond each capability's scope |

## Vendor Leakage and Boundary Debt

No `torch_fl` imports were found in user APIs, Simulation, or generic Runtime;
architecture rules permit them only in `compute/flagos.py`. No Torch-FL Python
objects were found entering `fq.Circuit`, FlagQuantum IR, or stable results.

Remaining boundary debt:

1. `runtime/distributed/flagos_runtime.py` directly reads
   `torch.flagos.set_device/current_device` outside the platform boundary.
   Eventually expose selection/current-device capabilities through `PlatformRuntime`;
   this round does not change that contract.
2. Statevector, MPS, TN, training-state, and operator-backend code directly calls
   `torch.cuda` stream/event/memory/RNG/synchronization APIs. Per-file ceilings in
   `architecture.toml` register the debt; unified PlatformRuntime calls have not
   replaced it.
3. Simulation uses `Tensor.is_cuda` and Triton CUDA restrictions for kernel
   selection. This leaks no CUDA Python objects but embeds vendor classification
   in numerics, limiting reuse by FlagOS/other PrivateUse1 platforms. Verified
   kernel capabilities should decide instead.
4. `compute/flaggems.py` reads and serializes FlagGems `vendor_name` as a string.
   No vendor object reaches user APIs, but the probe is outside PlatformRuntime's
   evidence model.
5. `PlatformRuntime.stream()`/`event()` return `Any`; `PlatformIdentity.metadata`
   and `PlatformDevice.metadata` accept arbitrary values. Implementations mostly
   return Torch objects or JSON-like scalars, but the contract does not prohibit
   unserializable vendor objects. Review this in the minimum contract.

`flagquantum/api.py` exposes generic `AcceleratorInfo`, without Torch-FL/Hygon/CUDA
SDK objects. `AcceleratorInfo` itself is Stable Core-related and unchanged this round.

## Minimum Compute Contract Proposal

The following is review input for Integration/Core, not a contract change on this
branch. Evolve existing `PlatformRuntime` instead of creating another Provider SPI.
Stability level, field names, and versioning require API/contract proposals.

| Group | Minimum fields/operations | Failure and evidence semantics |
| --- | --- | --- |
| Identity | `schema_version`, `provider`, `provider_version`, `vendor`, `device_type`, `runtime_versions`, `physical_device_id/name` | JSON-safe strings/scalars; unknown is null; identity does not promote capability |
| Discovery | `installed`, `activated`, `available`, `device_count`, `devices`, `availability_blockers` | Hints/mocks remain unverified, never available |
| Lifecycle | `activate`, `set_device`, `current_device`, `synchronize`, `stream`, `event`, `rng_state`, `restore_rng_state`, `close` | Optional capabilities explicitly unsupported; vendor objects remain adapter-private; controlled handles may reach Runtime |
| Memory | Per-device `allocated`, `reserved`, `free`, `total`, `peak_allocated`, `workspace`, `memory_kind` | Unmeasured is null; record sampling API, time, per-rank ownership |
| Kernels | `operation`, `dtype`, `layout`, `gradient`, `compiled/eager`, `supported`, `fallback_policy`, `probe_id` | Interface presence is not support; fallback requires authorization and result/evidence records |
| Precision | `storage_dtypes`, `compute_dtypes`, `accumulation_dtypes`, `native_fp64`, `native_complex128`, `software_precision_modes`, `precision_blockers` | Separate native from Double-Single; workload/kernel-specific evidence |
| Communication | `process_group_backends`, `p2p`, `collectives`, `dtype_support`, `device_resident`, `host_staging_observed`, `inner_backend`, `inner_backend_verified` | Logical backend differs from physical route; unknown blocks communication claims |
| Topology | `node_count`, `world_size`, `local_world_size`, `rank_placement`, `links`, `topology_source`, `topology_fingerprint` | Separate inference from measurement; unverified internode routes fail closed |
| Fallback | `requested_path`, `selected_path`, `fallback_allowed`, `fallback_used`, `fallback_reason`, `source_device`, `target_device` | No silent CPU/backend substitution |
| Evidence | `evidence_level`, `artifact_uri`, `artifact_sha256`, `code_revision`, `environment`, `workload`, `timestamp`, `blockers`, `claim_allowed` | Capability-specific; mocks, CPU semantics, and skips cannot promote hardware maturity |

First review priorities: move FlagOS set/current-device access into its owner;
define JSON-safe identity/device metadata; add unknown/unmeasured/unsupported/
verified semantics for kernels, precision, communication, and topology.
`DeviceExtension` may provide external extension lifecycles, but its frozen
`devices() -> Sequence[Mapping[str, Any]]` protocol cannot be tightened directly
on a team branch.

## Initial Conformance Test Scope

New tests verify only common invariants actually executable on CPU:

1. Built-in `cpu`, `cuda`, and `flagos` objects structurally satisfy `PlatformRuntime`.
2. CPU discovery/devices, identity, memory, RNG, events, streams, and synchronization
   are consistent and serializable.
3. A test adapter places CPU PlatformRuntime into the existing `ExtensionRegistry`
   device kind, verifying negotiation, start, invoke, cleanup, and task-local isolation.
4. No root export changes, new global registry, or optional FlagOS activation.

These tests neither execute nor simulate CUDA/FlagOS kernels, count skips as
evidence, or establish hardware claims. Real provider conformance requires each
hardware runner to produce an environment-bound payload with artifact digest,
then replacement verification through Integration-approved contract fakes and suites.

## Real Hardware Evidence Gaps

- At least one reproducible checked-in domestic physical-card P0–P5 result with
  provisioner attestation.
- Device name, driver, firmware, Torch-FL/FlagOS build, kernel origin, container digest.
- Per-kernel native FP64/complex128 support/error evidence and clear Double-Single boundaries.
- Observable absence of implicit CPU/CUDA fallback and host staging.
- FlagCX inner routes and complex collective matrix, particularly reduce-scatter.
- Domestic multi-GPU P2P bandwidth/correctness, collectives, rank placement, topology fingerprints.
- Domestic multinode network backends, links, timeout/failure recovery, collective correctness.
- Training evidence preserving one distribution semantics across sharded forward,
  backward, and optimizer updates.
- Peak device memory, workspace, communication bytes, per-rank ownership,
  single-device capacity failure.
- Repeated runs, determinism, long-running stability, performance, convergence,
  release-payload audits.

Until these close, use: "FlagOS adaptation has development/portability evidence
on CUDA-backed NVIDIA A800; real domestic-card support remains unverified."
Do not shorten this to a claim that domestic compute is supported.

## Statevector Verification Orchestration Review (2026-09-06)

Compute still supplies local FlagOS statevector platform identity. Per-operator
probes, precision requirements, execution precision plans, and numerical
certification are workload admission policy owned by Runtime, not intrinsic
platform capabilities. Runtime unified formerly separate preflight and numerical
certification in one FlagOS-specific internal entry, reading platform identity
once and failing closed in operator-support-then-numerical-certification order.
No `PlatformRuntime` contract changed and no conclusion was generalized to other
devices/workloads. Real/complex dtype pairing now reuses Backend Registry
`resolve_dtype()` instead of a second alias table.

## Device Resolution Boundary Review (2026-09-06)

Runtime first checks declared device types in `BackendCapabilities.devices`.
Built-in Platform Providers then activate, check availability, and validate device
indices. FlagOS remains an explicit-request special case to avoid loading
Torch-FL for backend summaries. Custom-backend-declared PyTorch device types without
registered built-in Compute support may still be parsed by `torch.device`.

Unknown-platform handling now catches only `KeyError` from the registry lookup
itself. Once a Compute implementation exists, activation/discovery `KeyError`
propagates; it cannot bypass Provider validation through bare `torch.device`.
This tightens ownership, not device substitution or CPU fallback.

Built-in PyTorch capability construction also reuses `with_accelerators()` instead
of duplicating device-list/preference projection. Environment discovery, injected
tests, and default capabilities share one rule: include only explicitly available
device types; only CUDA may become the automatic preference; FlagOS remains explicit.
