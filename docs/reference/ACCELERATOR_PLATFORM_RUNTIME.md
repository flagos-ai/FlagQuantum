# Accelerator Platform Runtime

This reference records the first implementation slice of the
[domestic accelerator and numerical trust plan](../roadmap/DOMESTIC_ACCELERATOR_AND_NUMERICAL_TRUST_PLAN.md)
and the
[FlagQuantum–Torch-FL integration plan](../roadmap/FLAGQUANTUM_TORCH_FL_INTEGRATION_PLAN.md).
It describes implemented interfaces, not a claim that any domestic accelerator
has passed production certification.

## Dependency rule

`flagquantum` continues to depend on PyTorch, not Torch-FL. Importing
FlagQuantum, querying backends, or running CPU/CUDA must not import `torch_fl`.
The optional provider is activated only when `flagos` is explicitly selected.
An absent or incompatible Torch-FL installation fails at activation with a
diagnostic error; it does not disable CPU or CUDA.

All domestic accelerator ownership remains in Torch-FL. FlagQuantum does not
identify or dispatch Hygon DCU, Ascend, GCU, MUSA, MetaX, or another vendor by
device name. Even when a vendor runtime is CUDA-compatible, Torch-FL owns that
compatibility route and exposes it through the `flagos` contract. FlagQuantum
records Torch-FL runtime identity and route evidence without duplicating vendor
branches.

```text
representation runtime
        |
        v
PlatformRuntime protocol
   |       |       |
  CPU     CUDA   FlagOS adapter --lazy--> Torch-FL
```

The architecture checker freezes the current direct `torch.cuda` call count in
legacy modules. New CUDA calls belong in the CUDA platform adapter. Reducing
the legacy counts is allowed; increasing them fails the architecture gate.
Direct `torch_fl` imports outside the FlagOS adapter also fail that gate.

## Hygon without hardware

Hygon preparation can be completed without a local DCU at the contract layer:

- mock `torch.flagos` discovery and `flagos:0` resolution;
- accept Torch-FL RuntimeIdentity metadata such as `vendor=hygon` and an
  internal `cuda_compatible` route;
- verify lazy activation, dependency isolation, error behavior, route policy,
  operator-profile schema, and CUDA non-regression;
- keep every quantum algorithm free of Hygon/DCU branches.

This is source and integration-contract readiness, not hardware certification.
Real DCU CI is still required for operator residency, complex128 behavior,
gradient correctness, convergence, distributed collectives, and performance.

## CUDA reference environment

Torch-FL's CUDA boxing design intentionally uses the PyTorch 2.10 CPU control
package and preloads a version-matched external `libtorch_cuda.so`. The `+cpu`
package label therefore describes the Python/ATen host package; it does not by
itself mean that `flagos:0` executed the workload on the CPU. FlagQuantum still
requires device residency, device memory, operator, gradient, and numerical
evidence from the logical FlagOS device.

The exact development harness is frozen in
[`ci/flagos_cuda_reference.lock.json`](../../ci/flagos_cuda_reference.lock.json).
It pins the container digest, Python, CUDA, PyTorch control package and CUDA
assets, Torch-FL commit, profile, dtypes, depths, and claim boundary. This lock
is a reproducibility contract for the reference lane, not a supported end-user
installation or a dependency of `pip install flagquantum`.

Torch-FL's CUDA backend provides the executable reference path. In a fresh
Torch-FL CUDA environment, run:

```bash
python tools/validate_flagos_cuda_reference.py
```

or enable the opt-in integration test:

```bash
FLAGQUANTUM_TEST_FLAGOS_CUDA=1 pytest tests/test_flagos_cuda_reference.py -v
```

The validator imports Torch-FL before PyTorch as required by Torch-FL's CUDA
distribution, resolves `flagos:0`, checks complex batched matrix multiplication,
scans complex64/complex128 circuits at depths 8/32/128 against a CPU complex128
reference, verifies the numerical-contract gradient evidence, checks logical
device residency, and synchronizes through
`FlagOSPlatformRuntime`. Its result explicitly remains
`hardware_certification=false`; it proves the joint FlagOS integration path,
not Hygon hardware quality or performance.

The CUDA reference was executed successfully on 2026-08-21 with an NVIDIA A100,
Torch-FL commit `2e00b393cf80088706b460a187aef185d3a283f4`, and PyTorch
2.10.0 CUDA 13.0 assets. All 36 selected tests passed, all 21 P0 requirements
were observed, and the complex64/complex128 depth scan passed at depths
8/32/128. The complete immutable result is recorded in
[`artifacts/flagos_cuda_reference_a100_20260821.json`](../../artifacts/flagos_cuda_reference_a100_20260821.json).
This remains CUDA reference evidence rather than domestic-card certification.

The manually triggered
[`FlagOS CUDA Reference`](../../.github/workflows/flagos-reference.yml) workflow
runs on a separately provisioned `flagos-cuda-reference` runner. It deliberately
does not install or mutate Torch-FL during the job: the Torch-FL team owns that
versioned runner image, while FlagQuantum verifies it against the lock before
accepting evidence. The job runs bounded focused tests, emits a raw JSON
artifact, and applies
[`tools/validate_flagos_reference_evidence.py`](../../tools/validate_flagos_reference_evidence.py).
The ordinary CPU CI also validates the checked-in artifact so documentation or
capability changes cannot silently promote it into hardware certification.

## Implemented contracts

- `PlatformRuntime` owns device discovery, synchronization, memory, stream,
  event, RNG state, profiler metadata, and runtime identity.
- FlagOS stream/event creation uses PyTorch's device-agnostic
  `torch.Stream(device=...)` and `torch.Event(device=...)` APIs, which dispatch
  through Torch-FL's registered PrivateUse1 guard without vendor branches.
- `CPUPlatformRuntime` and `CUDAPlatformRuntime` wrap current PyTorch behavior.
- `FlagOSPlatformRuntime` is a thin lazy adapter and contains the only Torch-FL
  activation boundary.
- `CapabilityEvidence` and `OperatorProfile` implement atomic, versioned,
  fail-closed operator preflight. Device names and environment hints are not
  accepted as verified evidence.
- The packaged [`statevector_local_p0` profile](STATEVECTOR_OPERATOR_PROFILES.md)
  is enforced before local statevector execution on `flagos`; accepted evidence
  is attached to the execution plan.
- The separate `split_real_imag_statevector_p0` profile gates an explicit,
  forward-only experimental executor whose state and gate arithmetic use two
  FP32 tensors. It is never selected by the default runtime.
- `FallbackPolicy`, `RouteExplanation`, `FallbackEvent`, and
  `StrictExecutionScope` make portable routes, host transfers, and dtype
  demotion explicit.
- `AccuracyRequirementContract` describes the numerical result required before
  planning. The name avoids collision with the existing post-execution
  `AccuracyContract` observation record.
- `PrecisionPlanContract` records independent parameter, gate, storage,
  compute, reduction, decomposition, gradient, optimizer, communication, and
  checkpoint dtypes plus complex representation and refinement strategy.
- The `statevector_local_p0` execution gate binds both contracts to a cached
  differentiable CPU-complex128 certification report. Unsupported plans fail
  closed before the user workload is allocated.

## CUDA compatibility invariant

CUDA remains the only accelerator selected automatically in this foundation
slice; FlagOS requires explicit selection until workload evidence is wired into
the profile resolver. CUDA discovery
continues to use `torch.cuda.is_available()`, `device_count()`, device names,
and memory properties through the adapter. No Torch-FL import or FlagOS probe
occurs on the CUDA path. CUDA correctness and performance baselines remain
mandatory release gates; this implementation does not replace CUDA kernels or
change their numerical dtype semantics.

## Current maturity boundary

The code now provides the Phase 0–2 control-plane foundation. It does **not**
yet certify statevector, MPS, tensor-network, density, gradient, distributed,
or high-precision execution on a domestic accelerator. Those capabilities can
be promoted only after all of the following exist for a named hardware/software
tuple:

1. a reviewed P0 operator profile;
2. executable forward/backward probes and evidence IDs;
3. CPU complex128 numerical reference comparisons;
4. a selected precision plan and explicit fallback policy;
5. real-device convergence and residency evidence;
6. CUDA non-regression evidence.

The split real/imaginary FP32 forward candidate and numerical acceptance suite
now exist behind an experimental entrypoint. The next implementation slice is
target-card route and residency evidence for that exact profile, followed by
parameter gradients and an accuracy monitor. Promotion still requires the
target card, Torch-FL/PyTorch version matrix, and hardware CI owner to be agreed
with the Torch-FL team.
