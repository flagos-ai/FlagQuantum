# Mapping Platform Facts to TargetCapabilities

Status date: 2026-09-06
Responsible team: Compute
Baseline: `vnext-phase1-contract-foundation`

## Conclusion

`flagquantum.compute.PlatformRuntime` can observe provider identity/version,
process-local installed/activated/available state, enumerated devices/names, some
total and allocator memory, and whether stream/event/RNG/synchronization handles
are callable. It has no fact fields for dtype, native double precision,
Double-Single, kernel residency, topology, P2P, collectives, internode communication,
or CPU fallback.

Platform cannot promote `_compiler.TargetCapabilities`, `BackendCapabilities.dtypes`,
extension `CapabilityResponse`, method presence, environment variables, or vendor
metadata into hardware support. Initial Platform-to-Core projection preserves
three orthogonal axes: `support_status`, `fact_exposure`, `evidence_level`. Missing
fields become `unknown/not_exposed` with blockers, not false, zero, empty topology,
or assertions of no fallback.

This slice adds Platform-owned `cpu_platform_to_target_capability_snapshot`, a
narrow adapter accepting injected probes, time, and evidence. It introduces no
Provider registration system, changes no Platform API/defaults, and touches no
protected Core/API contracts. Subsequent CUDA projection is restricted to the
actually passed single-card `statevector_local_p0/complex128` scope, without
extrapolation to other workloads, multicard execution, communication, or absence
of CPU fallback.

Classification: `single_device_fast_path` discovery characterization. Cited sharding/
communication records describe existing evidence boundaries, not new hardware or
scalability evidence.

## Three Independent Dimensions

| Dimension | Values | Question answered | Fail-closed rule |
| --- | --- | --- | --- |
| Support status | `unknown` / `unmeasured` / `unsupported` / `verified` | Does this target support this capability? | Only observations matching target/device/dtype/kernel/workload scope qualify as verified; declarations are at most unmeasured |
| Fact exposure | `observed` / `declared` / `not_exposed` / `unknown` / `not_applicable` | Where did the value come from, or why is it absent? | Missing SDK field: not_exposed; unclear origin: unknown; not_applicable requires justification |
| Evidence level | `basic` / `observable` / `certification` | What is the evidence ceiling? | Identity, version, discovery, calls, vendor declarations default to basic; workload-bound observations may reach observable; approved matrices/audits are required for certification |

Support status describes capability, not whether the probe call succeeded.
An explicit `available=false` observation is `unsupported/observed/basic`;
`available=true` is `verified/observed/basic`. Neither lifecycle conclusion
propagates to dtype, kernels, communication, or absence of fallback.

### Fact Exposure Reconciliation

| Exposure | Acceptable sources/examples | Possible support status | Constraint |
| --- | --- | --- | --- |
| `observed` | Lifecycle probes, memory API values, workload-bound operator/route probes | `verified` or `unsupported` | Bind positive and negative conclusions to actual scope; observed failures are not unknown |
| `declared` | JSON-safe provider metadata, backend dtype candidates, external attestations | Default `unmeasured` | Provider availability cannot promote declarations |
| `not_exposed` | CPU memory; missing Platform dtype/topology/P2P/collective/residency/fallback fields | `unknown` | After Core nullable-fact support, emit null with unknown/not_exposed and nonempty blockers; never false/zero/empty sets |
| `unknown` | Unclear sources, unserializable SDK objects, conflicting/unattributed routes | `unknown` | Discard vendor objects; retain portable blockers |
| `not_applicable` | Collective facts in a snapshot explicitly scoped to `world_size=1`, for example | `unknown` | Requires applicability reasons; not a substitute for unsupported/unmeasured |

Also distinguish:

- **API model support:** vocabulary/types/methods can express a capability; devices
  may still be unavailable.
- **Current environment availability:** providers/devices were discovered in this
  process; a dtype/kernel/workload may still be unsupported.
- **Verified evidence:** results bound to provider, physical device, versions, code
  revision, workload, time, and digest; valid only within their scope.

## Currently Observable Fields

| Domain | Generic source | Observable fact | Default axes | Cannot infer |
| --- | --- | --- | --- | --- |
| Identity | `PlatformIdentity` | Provider, device type, Torch/provider versions, vendor string | Existing values: verified/observed/basic identity only | Hardware capability or domestic certification |
| Lifecycle | installed/activated/is_available | Current process state | True: verified/observed/basic; explicit false: unsupported/observed/basic | Kernel/dtype/communication support |
| Discovery | `discover()/PlatformDevice` | Logical device, index, name, availability | Enumeration: verified/observed/basic | Vendor classification or capability from names |
| Device memory | `PlatformDevice.memory_bytes`, `MemorySnapshot` | CUDA usually total/allocated/reserved/free; FlagOS depends on public SDK; CPU absent | Values: verified/observed/basic; absent: unknown/not_exposed/basic | Missing is not zero; no capacity, peak, workspace, or workload-fit guarantee |
| Stream/event | `stream()` / `event()` | Controlled calls return handles | API call observation only; independent semantics/timing/concurrency/residency facts needed | Timing, asynchronous execution, device residency; handles cannot enter snapshots |
| Dtype | Backend candidates, probes, artifacts | Backend candidates and observed operator/dtype cells | Registry declarations: unmeasured/declared; passed runtime probes: verified within scope | Universal device dtype support from labels/tensor types |
| Native double precision | No Platform field; hardware/operator records | Workload-specific float64/complex128 execution | Without records: unknown/not_exposed/basic; observed requires native-path attribution | All CUDA or domestic FlagOS FP64 from A800 complex128 |
| Software precision | Double-Single contracts/implementation/A800 artifacts | Restricted P0–P5 high/low FP32 representations | API/algorithm presence: unmeasured/declared; artifacts may promote exact scope | General FP64/complex128 equivalence or full autograd/torch.optim support |
| Kernel residency | Operator probes/workload evidence | Per-operator/dtype forward/backward or logical residency | Bind probe/profile/workload; Platform alone: unknown/not_exposed | No host kernels/fallback from availability, streams, or output labels |
| Topology/P2P | Runner rank placement/topology fingerprints | Artifact-recorded nodes, ranks, links | No current snapshot source: unknown/not_exposed/basic | P2P from device count; bandwidth/routes from fingerprints |
| Intranode collectives | Runtime conformance/artifacts | Backend/collective/dtype/world-size cells | Per-cell evidence; failures may be unsupported/observed; untested is unmeasured | All collectives from process-group initialization |
| Internode communication | Multinode runners/artifacts | Network/backend/topology/workload results | No matching record: unknown/not_exposed/basic | Multinode support from single-node NCCL/FlagOS |
| CPU fallback | Workload route audits/results | Requested/selected paths, authorization/use/reason | Actual observation: verified/observed/observable; hidden routes: unknown/not_exposed | `fallback_used=false` from not_exposed |

## Platform Fact Matrix

| Platform | API model | Currently observable | Verified scope | No guarantee |
| --- | --- | --- | --- | --- |
| CPU | Full lifecycle; backend lists complex64/complex128 | One CPU, sync, host events, null streams, RNG; memory absent | Paths covered by local numerical/training/operator tests | Platform CPU memory, NUMA, topology, communication, per-kernel native FP64, fallback facts |
| NVIDIA CUDA | Discovery, memory, streams/events, RNG; Runtime NCCL | Current devices/names/memory/allocator and CUDA handles | Devices/versions/dtypes/workloads/world sizes in checked-in A100/A800 artifacts | Every NVIDIA model/kernel/dtype, timing, P2P/multinode, universal no-CPU-fallback |
| FlagOS | Lazy Torch-FL lifecycle; Runtime backend=flagos | SDK-exposed devices, optional memory, streams/events/RNG, identity | Checked-in CUDA-backed Torch-FL on A800 development evidence | Domestic cards, FlagCX routes, no host staging, all collectives, multinode, production |
| Real domestic cards | Attested P0–P5 candidate harness | No checked-in real-card observations | No reviewed real-card result | Vendor/model/driver, native FP64, Double-Single, residency, communication, topology, no-fallback all unknown |
| Other vendors/hints | Extension/Platform names and `FLAGQUANTUM_ACCELERATOR` | Hints produce unavailable unknown accelerators | Mocks verify isolation/rejection | Any real hardware support |

### Precision Boundaries

`BackendCapabilities.dtypes=(complex64, complex128)` is an execution-model candidate
set, not per-device discovery. `CapabilityEvidence` verifies operator/dtype cells
only after passed `runtime_probe` or `hardware_ci`, never unprobed kernels.

Double-Single is software precision, independent of `native_fp64/native_complex128`.
P3/P4/P5 cover full-state high/low, bounded device gate generation, and explicit
Double-Single SGD respectively. Preserve FP32 exponent range, gate restrictions,
ordinary autograd FP32 delivery, and non-`torch.optim` limitations with these facts.

### Communication and Fallback Boundaries

PlatformRuntime has no process-group, P2P, collective, rank-placement, or topology
methods. `build_distributed_identity()` separates outer backend and inner route:
FlagOS initialization leaves inner backend, FlagCX, and host staging unknown and
disables communication claims. CPU fallback facts require execution observation
or audited provider attestation. Labels and successful results cannot establish
absence of host execution.

## Evidence Sources and Ceilings

| Source | Establishes | Does not establish |
| --- | --- | --- |
| `tests/unit/test_platform_runtime.py` | CPU lifecycle, CUDA discovery fakes, FlagOS lazy/public API adaptation, no vendor-name classification | Real CUDA/FlagOS/domestic execution |
| `flagquantum/compute/cpu_target_capabilities.py` | Injected CPU device/count/memory/precision projection, independent identity/scope, TTL/evidence propagation, missing-fact blockers | CUDA/FlagOS/QPU discovery, requirements/fallback, performance/hardware claims |
| `tests/team/compute/test_cpu_target_capabilities.py` | Source/evidence round trips, unavailable/missing/negative probes, TTL/scope, unchanged defaults | Hardware capability; fake probes are not hardware evidence |
| `cuda_target_capabilities.py` + `probe_cuda_target_capabilities.py` | Workload-bound snapshots after real single-card probes, statevector, numerical/gradient checks | No hidden CPU fallback, multicard/multinode, communication, performance, domestic compute |
| `artifacts/cuda_target_capabilities_a800_jp17{1,2}_20260906.json` | Comparable observable PyTorch CUDA complex128 evidence on one A800 per node; equivalent contracts/metrics after replacement | Certification or other devices/versions/workloads |
| `tests/team/compute/test_target_capability_facts.py` | Three-axis separation, missing fields unknown, no declaration promotion/object leakage | Hardware capability; test fixtures are not Core contracts |
| `runtime/operator_probes.py` + `CapabilityEvidence` | Provider/device/profile/operator/dtype forward/backward probes | Unprobed operators, communication, topology, physical routes, production level |
| `artifacts/flagos_cuda_reference_a800_20260824.json` | Single flagos:0 CUDA-backed Torch-FL reference on A800 | Domestic/native FlagOS hardware, no host fallback |
| `artifacts/flagos_workload_capability_f4_20260826.json` | Single-node 2/4/8-card FlagOS-on-A800 matrix and reduce-scatter gap | FlagCX routes, host staging, multinode, certification/release |
| `artifacts/flagos_statevector_capacity_f5_a800_20260827.json` | Scoped 32-qubit complex128 single/replicated OOM versus eight-rank completion | General/backward/optimizer capacity, domestic hardware, production scalability |
| `artifacts/flagos_transport_observability_f6_a800_20260827.json` | Single-node four-collective logical-device results for two complex dtypes | Inner transport, no host staging, performance, multinode |
| `artifacts/split_real_imag_*_a800_*.json` | Specified A800 CUDA/FlagOS-on-CUDA P0–P5 development paths | General FP64 equivalence, all gates/optimizers, convergence, domestic/production capability |
| `tools/validate_domestic_single_card.py` | Strict attestation/P0–P5/no-fallback candidate acceptance | Existing domestic-card evidence; candidate success is not hardware certification |

`a800-node-0` and `a800-node-1` are NVIDIA A800 nodes. Session availability or
transient runs cannot replace reviewed checked-in artifacts with digests/revisions.
Success on those nodes establishes only NVIDIA CUDA scope, not domestic hardware
or FlagOS evidence.

## Narrow CPU Adapter Implementation

`flagquantum.compute.cpu_target_capabilities` is the first implementation slice.
Callers inject `CPUCapabilityProbe`; `observe()` returns `CPUCapabilityObservation`.
They also supply independent target ID, provider version, target revision,
environment ID, probe `source_ref`, and static `target_class_source_ref`.
The adapter fixes `target_class=local_runtime`, `provider=pytorch_cpu`, and uses
observed device IDs for scope. It emits an authoritative static
`target.class=local_runtime` fact for Compiler/CPU matching.

- Observed available devices produce verified/observed `device.kind` and
  `device.count`. These facts, memory, and explicit precision observations require
  at least observable probe evidence; static target.class may use basic evidence.
- `available=false, device_count=0` emits unsupported/observed device.count=0 and
  `cpu_unavailable`, never substitution with another target.
- Missing memory/precision yields null, unknown/not_exposed, and nonempty blockers,
  not zero, empty strings, or invented dtypes. Core rejects based on fact status.
- Explicit unsupported/unmeasured precision preserves its status and blockers;
  declarations, Python dtypes, registry entries, and identity never become verified.
- Verified precision requires observed exposure; probe values reject verified/declared.
- Callers inject captured_at, positive TTL, and unchanged Core EvidenceReferences.
  The adapter invents no digests and retains no stream/event handles in snapshots.

The adapter produces snapshots only, not requirements, target selection, or Runtime
fallback. It does not affect `get_platform_runtime()`, registries, or default backends.

## Single-Card CUDA Observation Slice

`cuda_statevector_capability_snapshot` projects only passed single-device
`statevector_local_p0/complex128` evidence. The tool requires exactly one visible
CUDA card and real operator preflight, statevector execution, double-precision
numerical checks, and gradient comparisons before snapshot creation. Current A800
results are observable development evidence. Blockers explicitly retain unverified
hidden CPU fallback, multicard/multinode behavior, and production performance;
this is not general CUDA, FlagOS, or domestic certification.

Independent UUIDs/snapshot identities differ between `a800-node-0` and
`a800-node-1`, while precision facts, workload scope, metrics, and blockers match.
This is the first physical-node replacement verification without consumer changes.

## Minimum Platform-to-Core Projection Proposal

Review input only. Core owns final schemas, enums, and serialization. Platform
adapts current PlatformRuntime/SDK sources without introducing a registry.

```text
PlatformCapabilitySnapshotCandidate
  schema_version
  snapshot_id / observed_at
  provider_identity
    provider, provider_version, device_type, vendor
    runtime_versions, physical_device_id/name
  environment
    installed, activated, available, availability_blockers
  device
    logical_device, index, memory facts
  facts[name]
    value
    support_status
    fact_exposure
    evidence_level
    source_kind / source_ref
    scope {device, dtype, kernel, workload, world_size, node_count}
    blockers
  evidence_refs[] {artifact_uri, sha256, code_revision, environment_id}
  snapshot_blockers[]
```

Proposed initial fact names: `storage_dtypes`, `compute_dtypes`,
`accumulation_dtypes`, `native_fp64`, `native_complex128`,
`software_precision_modes`, `device_memory_*`, `stream_semantics`,
`event_semantics`, `kernel_residency`, `topology`, `p2p`,
`intra_node_collectives`, `inter_node_communication`, `cpu_fallback_used`.

Projection rules:

1. Platform identity remains identity, without capability promotion.
2. Provider metadata is JSON-safe declared/basic input only. Objects, callables,
   live handles, and unknown free-form fields stay outside Core snapshots.
3. Missing SDK fields yield null, unknown/not_exposed, and specific blockers.
   Snapshot-level blockers cover global unavailability only. Core rejects nullable
   nonverified facts.
4. available=true verifies environment availability only, not other domains.
5. Unsupported requires an applicable authoritative negative probe; unrun probes
   remain unmeasured/unknown.
6. Not-applicable requires reasons, never concealment of missing implementation/tests.
7. Observations bind source, scope, time; stale snapshots are not current facts.
8. Claim ceilings use the weakest required fact/evidence level. Unknown,
   unmeasured, not-exposed, or stale requirements fail preflight.
9. Vendor handles remain adapter-private, used through Core-owned opaque IDs or
   local callbacks. Unserializable objects stay outside Runtime results, Simulation
   state, and user APIs.
10. Unobserved CPU fallback with hidden routes retains unknown blockers, never false.
11. Core admits only required verified/observed, fresh facts with sufficient
    evidence. Unsupported, unmeasured, unknown, not-exposed, and unjustified
    not-applicable cases fail closed.

## Vendor Leakage Review

1. `runtime/distributed/flagos_runtime.py` directly accesses
   `torch.flagos.set_device/current_device` outside Platform. Move this under an
   approved device-selection contract later; this slice changes no contract.
2. Runtime/Simulation CUDA fast paths use registered per-file architecture ceilings.
   They impede platform replacement and are not generic snapshot sources.
3. `compute/flaggems.py` converts vendor_name to a string, without object leakage;
   its operator catalog is declaration/probe input, not hardware certification.
4. Stream/event returns and identity/device metadata still allow Any rather than
   contractual JSON safety. Torch/provider handles must stop at adapters;
   projections accept only scalars, closed string lists, controlled references.
5. No torch_fl imports were found in root user APIs or Simulation. Generic
   AcceleratorInfo does not carry Torch-FL/Hygon SDK objects.

## Currently Unproven

- Any verified/certified real domestic card.
- Domestic native FP64/complex128, complete Double-Single paths, kernel residency,
  or absence of CPU/CUDA fallback.
- FlagCX inner routes, no host staging, complete complex collectives/P2P.
- FlagOS multinode or domestic multi-GPU/multinode placement, topology, networking,
  fault tolerance.
- Workload support inferred only from availability, names, hints, mocks, interfaces.
- Unchecked-in transient remote results as release, production, certification, or
  domestic-compute evidence.

The accurate description remains: CPU is a real local path; NVIDIA CUDA has
scope-limited hardware evidence; FlagOS has adaptation development evidence on
CUDA-backed A800; real domestic-card capability is unknown and unverified.
