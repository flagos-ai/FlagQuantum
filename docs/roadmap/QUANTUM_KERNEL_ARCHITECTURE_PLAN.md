# Quantum Kernel Architecture Plan

> **Status:** Phases 0 and 1 complete in `a2f24e4`; Phases 2-5 proposed
>
> **Scope:** FlagQuantum accelerator kernels, Torch-FL integration, FlagTree
> compilation, CPU reference implementations, kernel capability evidence, and
> migration of `flagquantum/simulation/triton_kernels/`
>
> **Change boundary:** Phase 1 relocated private kernel modules without changing
> their algorithms or the stable public API. It did not add a mandatory
> dependency or claim support for an unverified accelerator.

## 1. Decision Summary

FlagQuantum introduces a first-class internal `flagquantum/kernels/` domain
before the FlagTree integration surface grows.

Phase 1 was a behavior-preserving relocation of the existing optional Triton
kernels from:

```text
flagquantum/simulation/triton_kernels/
```

to:

```text
flagquantum/kernels/triton/
```

The directory is not a second simulation framework. It owns optional,
capability-selected, optimized kernel implementations. Mathematical semantics,
portable PyTorch implementations, and CPU correctness references remain in
`flagquantum/simulation/`.

The target architecture has two complementary execution paths:

1. **Primary FlagOS path:** FlagQuantum uses Torch-FL for device ownership,
   streams, memory, autograd integration, distributed process groups, operator
   routing, and runtime identity. Torch-FL may route work to FlagTree, vendor
   kernels, compatibility boxing, FlagGems, or an explicitly reported fallback.
2. **Auxiliary quantum-kernel path:** FlagQuantum owns quantum-specific Triton
   kernels and may compile them through a direct Triton route or FlagTree. This
   path does not replace Torch-FL's device responsibilities. On a `flagos`
   device, it must use the device and stream contract established by Torch-FL.

FlagQuantum should not create a second multi-vendor device runtime, duplicate
Torch-FL vendor branches, or treat FlagTree as the quantum circuit compiler.
FlagTree is a numerical kernel compiler; the FlagQuantum circuit compiler and
FlagQuantum IR remain separate authorities.

## 2. Why This Boundary Is Needed Now

Before Phase 1, `flagquantum/simulation/triton_kernels/` contained specialized
implementations for several distinct numerical domains:

- complex batched matrix multiplication;
- statevector gate application;
- MPS two-site contraction;
- repeated single-qubit layers;
- HVA forward tangents; and
- two-qubit Pauli tangents.

These modules are no longer a small local detail of one simulation algorithm.
They form an optional accelerator-kernel subsystem used across statevector,
MPS, contraction, and differentiable execution paths. Keeping that subsystem
inside `simulation/` obscures ownership and makes future FlagTree provider,
capability, provenance, and backend work harder to review.

The move precedes the addition of:

- FlagTree compiler identity and backend evidence;
- `flagos` tensor support for quantum-specific kernels;
- kernel capability manifests;
- TLE-specific implementations;
- distributed compute/communication fusion kernels; or
- optimized CPU-native kernels.

Moving later would mix a structural change with compiler and device behavior
changes, increasing regression and review risk.

## 3. Goals

1. Give optimized quantum kernels one clear internal owner.
2. Preserve current portable CPU and PyTorch execution paths.
3. Keep Triton and FlagTree optional and lazily imported.
4. Allow runtime policy to distinguish PyTorch, direct Triton, and FlagTree.
5. Record device runtime and kernel compiler identity separately.
6. Support native CUDA and Torch-FL `flagos` tensors without duplicating device
   runtime ownership.
7. Make fallback, dtype demotion, host transfer, and unsupported capabilities
   explicit and auditable.
8. Organize kernels by quantum semantics rather than accelerator vendor.
9. Permit future optimized CPU kernels without moving ordinary CPU reference
   algorithms out of `simulation/`.
10. Keep the stable user surface centered on `fq.Circuit`, `fq.Module`,
    `fq.plan`, `fq.run`, and training workflows.

## 4. Non-Goals

This plan does not:

- make Torch-FL, Triton, or FlagTree a mandatory core dependency;
- create public kernel APIs in the initial migration;
- create a separate `FlagQuantum-Kernels` repository;
- move all numerical code into `flagquantum/kernels/`;
- move ordinary CPU/PyTorch reference implementations;
- add vendor-specific quantum algorithm branches;
- promise every Triton kernel works on every FlagTree backend;
- infer hardware support from package installation or import success;
- permit silent CPU fallback or silent dtype demotion;
- merge the FlagQuantum circuit compiler with the FlagTree kernel compiler; or
- change capability maturity without workload-specific evidence.

## 5. Ownership Model

### 5.1 Simulation

`flagquantum/simulation/` owns mathematical and numerical semantics:

- statevector, MPS, tensor-network, density-matrix, and noise algorithms;
- portable PyTorch implementations;
- CPU `complex128` correctness references;
- backend-neutral gate and observable semantics;
- reference gradients and finite-difference checks;
- precision algorithms such as split real/imag and Double-Single; and
- deterministic fallback implementations.

Simulation answers: **What calculation is mathematically required?**

### 5.2 Kernels

`flagquantum/kernels/` owns optional optimized implementations:

- Triton and FlagTree-compatible Triton kernels;
- optional TLE extensions;
- future C++/SIMD/OpenMP CPU kernels;
- launch wrappers and custom autograd wrappers tightly coupled to a kernel;
- kernel-local shape and layout validation; and
- implementation-specific tuning parameters.

Kernels answer: **How can a proven mathematical operation execute efficiently
on a supported target?**

### 5.3 Runtime

`flagquantum/runtime/` owns execution policy:

- kernel-provider discovery and selection;
- capability preflight;
- strict fallback policy;
- execution-plan binding;
- compiler and route evidence;
- distributed ownership and collectives;
- failure classification; and
- public result assembly.

Runtime answers: **Which authorized implementation should execute this
operation in this environment?**

### 5.4 Platform Runtime

The existing platform boundary owns directly controlled device resources:

- device discovery and activation;
- memory information;
- streams and events;
- synchronization;
- random-number-generator state;
- profiler metadata; and
- device runtime identity.

For `device="flagos"`, Torch-FL remains the authority. FlagQuantum must not
duplicate Hygon, Ascend, MUSA, MetaX, GCU, or other vendor runtime branches.

### 5.5 Compiler Identities

Two compiler identities must remain distinct:

1. **Circuit compiler identity:** FlagQuantum IR version, circuit legalization,
   optimization, routing, scheduling, and target-text generation.
2. **Kernel compiler identity:** the measured compiler distribution, FlagTree,
   or another numerical kernel compiler, including its version and selected
   backend.

Kernel compiler facts must not be inserted into the circuit compiler identity
as if both compilers were one pipeline.

## 6. Target Architecture

```text
fq.Circuit / fq.Module / fq.run
                |
                v
        FlagQuantum IR and planner
                |
                v
        Representation runtime
        /                   \
       /                     \
      v                       v
Simulation reference     Kernel dispatch
(portable PyTorch)        and capability gate
                              |
                       +------+------+
                       |             |
                       v             v
                 direct Triton    FlagTree
                       |             |
                       +------+------+
                              |
                              v
                  current tensor and stream
                       /               \
                      v                 v
               native CUDA       Torch-FL `flagos`
                                      |
                                      v
                         accelerator/vendor runtime
```

The two intended product paths are:

```text
Primary:
FlagQuantum -> Torch-FL -> FlagTree / FlagGems / vendor kernel / boxing

Auxiliary:
FlagQuantum quantum kernel -> Triton-compatible compiler -> current device
```

The auxiliary path bypasses Torch-FL's generic operator routing when it launches
a FlagQuantum-owned kernel. It does not bypass Torch-FL's ownership of a
`flagos` device, stream, allocator, or runtime registration.

## 7. Proposed Directory Structure

The initial structure should remain small:

```text
flagquantum/
├── kernels/
│   ├── __init__.py
│   └── triton/
│       ├── __init__.py
│       ├── _jit.py
│       ├── complex_bmm.py
│       ├── statevector_gates.py
│       ├── statevector_adjoint.py
│       ├── mps_two_site.py
│       ├── single_qubit_loop.py
│       ├── hva_forward_tangent.py
│       └── two_qubit_pauli_tangent.py
├── simulation/
├── runtime/
└── ...
```

Subdirectories should be introduced only when the number of modules justifies
them. A likely later structure is:

```text
flagquantum/kernels/
├── triton/
│   ├── statevector/
│   ├── mps/
│   ├── tensor_network/
│   ├── numerics/
│   └── extensions/
│       └── tle/
├── cpu/
│   ├── cpp/
│   └── simd/
└── manifests/
```

The repository should not create parallel source trees named `triton/` and
`flagtree/` for the same kernels. FlagTree provides a Triton-compatible compiler;
shared kernel sources belong under `triton/`. FlagTree-specific identity,
capability, and optional TLE behavior belong in provider metadata or a narrow
extension directory.

## 8. CPU Policy

`flagquantum/kernels/` includes accelerator kernels, but it is not defined as a
GPU-only package. The boundary is based on implementation role, not hardware
name.

### 8.1 CPU Code That Remains in Simulation

The following stays in `flagquantum/simulation/`:

- ordinary PyTorch tensor operations;
- CPU statevector, MPS, and tensor-network algorithms;
- `complex128` correctness references;
- portable `torch.einsum` and `torch.matmul` implementations;
- finite-difference and reference-gradient logic;
- host-side Double-Single reference arithmetic; and
- deterministic fallback implementations.

These paths are the semantic authority and test oracle. They must remain easy
to install and run without Triton, FlagTree, Torch-FL, or a native extension.

### 8.2 CPU Code That May Enter Kernels

Future CPU implementations may live under `flagquantum/kernels/cpu/` only when
they are separately optimized and independently dispatched, for example:

- AVX2 or AVX-512 statevector kernels;
- C++/OpenMP gate application;
- a custom PyTorch C++ extension;
- oneDNN/MKL specialization;
- tuned bit-permutation kernels; or
- bounded and measured CPU compiled kernels with their own cache policy.

No CPU directory should be created merely for symmetry.

## 9. Kernel Provider Contract

The current dispatch distinction between `pytorch` and `triton` is too coarse
for the target architecture. A future internal provider contract should expose
at least:

```python
class KernelProvider(Protocol):
    def identity(self) -> KernelProviderIdentity: ...
    def supports(self, request: KernelRequest) -> KernelCapabilityResult: ...
    def select(self, request: KernelRequest) -> KernelDecision: ...
```

Initial provider identities should distinguish:

- `pytorch_eager`;
- `torch_compile`;
- direct Triton (`compiler_distribution="triton"`, `integration_path="direct"`);
  and
- `flagtree`.

Importability is not capability. `find_spec("triton")` can establish only that a
module is present. It does not establish compiler identity, active backend,
primitive support, device launch compatibility, gradient correctness, dtype
support, or absence of fallback.

Provider selection must consume explicit capability evidence rather than infer
support from a vendor name or version threshold alone.

## 10. Kernel Organization Rules

Kernels should be organized by quantum or numerical responsibility:

```text
statevector/local_1q
statevector/local_cx
statevector/adjoint_vjp
mps/two_site
tensor_network/pair_contract
numerics/complex_bmm
```

They should not be organized primarily by vendor:

```text
# Avoid
kernels/nvidia/
kernels/hygon/
kernels/ascend/
kernels/mthreads/
```

Vendor directories in FlagQuantum would duplicate FlagTree and Torch-FL
responsibilities. A backend-specific specialization is acceptable only when a
capability difference cannot be expressed through shared source, tuning, or a
narrow extension. It must have an owner, evidence, a portable reference, and a
documented removal or convergence condition.

Each maintained optimized kernel should eventually have four associated
artifacts:

1. implementation and launch wrapper;
2. correctness and gradient tests;
3. a machine-readable capability manifest; and
4. a reproducible benchmark workload.

## 11. Capability and Evidence Model

A kernel capability request should include enough information to reject an
unsafe launch before allocating a large workload:

```text
kernel semantic ID
device type and index
device runtime provider
kernel compiler provider and backend
dtype and precision plan
layout and shape constraints
forward/backward requirement
distributed semantics
required compiler primitives
fallback authorization
```

Execution evidence should keep platform, compiler, and route identities
separate. A representative direct-Triton record is:

```json
{
  "device_runtime": {
    "provider": "pytorch",
    "device_type": "cuda"
  },
  "kernel_compiler": {
    "distribution": "triton",
    "version": "<measured>",
    "backend": "cuda",
    "identity_source": "python_package_metadata",
    "identity_status": "resolved"
  },
  "kernel_route": {
    "semantic_id": "statevector.local_1q",
    "implementation": "triton",
    "integration_path": "direct",
    "fallback": false
  }
}
```

`direct` means FlagQuantum called the measured Triton distribution without a
FlagTree integration layer. It does not claim that the distribution is an
unmodified upstream build. A future FlagTree route should report
`integration_path="flagtree"` and FlagTree's measured integration version in
addition to the underlying compiler facts.

Because FlagTree installs a `flagtree` distribution that provides the
`triton` Python module, provenance must come from the module-to-distribution
mapping rather than from `import triton` or a lookup of the `triton`
distribution name alone. Missing, ambiguous, or unsupported ownership is
reported as `integration_path="unknown"`; it must not be relabeled as direct
Triton or FlagTree from a device name or environment hint.

The record must not claim a FlagTree route merely because Torch-FL is active.
Torch-FL can select vendor-native kernels, compatibility boxing, FlagGems,
FlagTree, or fallback paths. The actual route must be measured or supplied by
an authoritative runtime/compiler interface.

### Current `statevector.local_1q` user path

Users do not call the Triton launch wrapper directly. They submit a statevector
circuit containing one-qubit gates to the distributed runtime, and the executor
selects the kernel when the feature is requested and the input is eligible:

```bash
FQ_STATEVECTOR_TRITON_LOCAL_1Q=1 \
  python -m examples.triton_statevector_local_1q
```

The complete example builds the circuit with `fq.Circuit`, executes it through
the public `flagquantum.runtime.run_distributed` entry point, and reads the
native result summary to verify the selected route. The ordinary single-GPU
`fq.run(..., mode="statevector")` path currently uses the local simulator and
does not call this distributed-statevector kernel. The kernel requires a
contiguous CUDA `complex64` statevector, an available `triton` module, a
non-portable runtime mode, and a one-qubit gate on a local qubit. Every qubit is
local on a one-rank run. With a sharded statevector, a qubit is local only when
both amplitudes in each affected pair are owned by the same rank.

## 12. Fallback Rules

- `mode="auto"` may authorize initial selection; it does not automatically
  authorize fallback after a failed launch.
- Unsupported dtype, layout, gradient, or compiler primitives must fail
  capability preflight or select a declared portable implementation.
- CPU transfer, dtype demotion, and host staging must never be silent.
- A failed optimized kernel must not repeatedly recompile on every invocation.
- Evidence must identify optimized and fallback operation counts.
- A fallback result cannot be cited as evidence for the rejected kernel route.

## 13. Dependency and Import Rules

The following rules should be added to executable architecture policy when the
directory is introduced:

1. `flagquantum.kernels` may depend on PyTorch and optional kernel-language
   packages.
2. Importing `flagquantum` must not import Triton, FlagTree, Torch-FL, vendor
   SDKs, or compiled kernel extensions.
3. `flagquantum.kernels` must not import `flagquantum.runtime`.
4. `flagquantum.kernels` must not import Torch-FL directly.
5. `flagquantum.kernels` must not select devices or distributed topology.
6. Runtime may import kernel launch wrappers only after provider and capability
   selection.
7. Simulation may use backend-neutral kernel contracts but must not own runtime
   selection policy.
8. Torch-FL imports remain confined to the approved FlagOS platform adapter.
9. CPU and native PyTorch paths must work when optional kernel dependencies are
   absent.

## 14. Migration Plan

### Phase 0: Approve the Boundary (Complete)

- Review this document with Runtime, Simulation, Torch-FL, and FlagTree owners.
- Confirm `flagquantum/kernels/` is internal in the first release.
- Confirm no stable public API change is included.
- Add the package and dependency direction to `architecture.toml`.

Exit criterion: ownership and import direction are agreed before code moves.

### Phase 1: Behavior-Preserving Directory Move (Complete)

- Create `flagquantum/kernels/triton/`.
- Move existing Triton modules without algorithm changes.
- Update internal imports and focused tests.
- Preserve lazy imports and optional dependency behavior.
- Update all repository-owned private imports rather than retaining the old
  private package as a second source location.
- Do not generalize CUDA checks in this phase.

Exit criteria:

- CPU import and test paths do not import Triton;
- CUDA behavior and kernel-selection evidence are unchanged;
- no stable root exports change;
- architecture and dependency-policy tests pass; and
- old and new numerical outputs match existing references.

Current state: `a2f24e4` moved the modules as a pure rename into
`flagquantum/kernels/triton/`, and `flagquantum/simulation/triton_kernels/` no
longer exists. No repository-owned import still names the old path.

### Phase 2: Kernel Provider and Identity

- Generalize statevector-local dispatch into a reusable internal provider
  contract.
- Distinguish PyTorch eager, `torch.compile`, direct Triton, and FlagTree.
- Record kernel compiler identity separately from circuit compiler identity.
- Add strict fallback and compile-failure caching.
- Add provider capability tests that do not require hardware.

Exit criterion: a result can state which compiler and implementation executed
without inferring identity from the device name.

### Phase 3: FlagTree on Native CUDA

- Run the same kernel sources through direct Triton and FlagTree's NVIDIA backend
  in separate pinned environments.
- Validate forward values, gradients, device residency, compiler identity,
  compilation cache behavior, and performance.
- Preserve PyTorch reference comparisons.

Exit criterion: the auxiliary FlagTree path is proven independently of
Torch-FL device adaptation.

### Phase 4: FlagTree on Torch-FL `flagos`

- Select one small split-real/imag or local gate kernel.
- Replace CUDA-name checks with capability checks for that bounded kernel.
- Obtain the current device and stream through the Torch-FL-owned platform
  contract.
- Validate no host transfer, no dtype demotion, forward and backward
  correctness, and exact compiler/backend identity.
- Start with one named hardware/software tuple.

Exit criterion: one complete
`FlagQuantum -> Torch-FL device -> FlagTree kernel -> accelerator` slice passes
workload-specific conformance on real hardware.

### Phase 5: Controlled Expansion

- Add complex BMM, additional statevector gates, adjoint kernels, and MPS
  kernels one at a time.
- Add distributed kernels only after single-device correctness and route
  evidence are established.
- Introduce TLE only for measured needs portable Triton cannot satisfy.
- Promote capability maturity per kernel/backend tuple, never per package.

## 15. Verification Matrix

| Lane | Purpose | Claim boundary |
| --- | --- | --- |
| CPU PyTorch | Semantic reference and import isolation | Correctness only |
| Native CUDA PyTorch | Existing accelerator non-regression | Native CUDA path |
| Direct Triton on CUDA | Baseline optimized kernel behavior | Measured direct Triton tuple only |
| FlagTree NVIDIA | Direct compiler-path comparison | FlagTree NVIDIA only |
| Torch-FL on CUDA | Main device integration reference | Torch-FL CUDA reference |
| Torch-FL + FlagTree on one domestic accelerator | End-to-end primary plus auxiliary path | Exact pinned tuple only |

Every optimized-kernel lane should cover, as applicable:

- forward and backward numerical comparison;
- dtype and precision-plan enforcement;
- device residency and stream correctness;
- deterministic or bounded numerical behavior;
- compile cache behavior;
- fallback absence or explicit fallback evidence;
- compiler and backend identity; and
- warmed-up performance reported separately from compile time.

## 16. Packaging Strategy

The first implementation remains in the main FlagQuantum distribution with
lazy optional imports. A separate repository or wheel is premature.

A future package such as `flagquantum-kernels` may be considered only if:

- kernels have an independent maintainer and release cadence;
- multiple external quantum frameworks consume them;
- optional compiler dependencies materially burden core installation;
- the hardware CI matrix dominates the main repository's release process; or
- the kernel ABI and provider contract are stable enough for independent
  versioning.

Until then, a separate repository would add version skew, release coordination,
CI duplication, and issue-routing cost without solving the current ownership
problem.

## 17. Risks and Mitigations

### The Move Becomes a Hidden Rewrite

Keep Phase 1 behavior-preserving. Do not combine relocation with provider
abstraction or backend generalization.

### `kernels/` Duplicates `simulation/`

Simulation owns semantics and references; kernels own optional optimized
implementations only.

### FlagQuantum Reimplements Torch-FL

Do not add vendor runtime branches, allocators, stream implementations,
PrivateUse1 registration, or generic operator routing to `kernels/`.

### FlagTree Installation Is Mistaken for Capability

Require per-kernel capability and execution evidence. Import and version checks
are diagnostics, not certification.

### Kernel Sources Fork Per Vendor

Organize by semantic operation, use shared Triton sources and capability-driven
specialization, and require explicit justification for backend-specific code.

### CPU Becomes a Second-Class Path

Keep CPU/PyTorch references in Simulation, keep optional imports lazy, and
retain CPU correctness as a release gate.

## 18. Open Questions

1. Should the provider contract live in `flagquantum/runtime/kernels/` or in a
   dependency-light internal contracts module?
2. Which Torch-FL public interface should provide authoritative per-operation
   route evidence?
3. Which FlagTree public marker should identify the distribution and backend
   without relying on unstable internals?
4. What is the first domestic hardware/software tuple for the end-to-end slice?
5. Which initial quantum kernel has the smallest primitive set and clearest
   gradient reference?
6. Should capability manifests ship as TOML package data or typed Python values
   with generated documentation?

## 19. Acceptance Criteria

This plan is ready for implementation when reviewers agree that:

- `flagquantum/kernels/` owns optional optimized quantum kernels;
- ordinary CPU/PyTorch references remain in `simulation/`;
- Torch-FL remains the `flagos` device-runtime authority;
- FlagTree is a kernel compiler, not the circuit compiler;
- the primary and auxiliary paths are complementary, not competing device
  stacks;
- Phase 1 contains no behavior or stable API change;
- kernel capability is evidence-driven and fail-closed; and
- the initial work stays in the main repository and package.
