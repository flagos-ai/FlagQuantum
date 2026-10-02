# FlagQuantum kernel catalog

`flagquantum.kernels` owns quantum-computing operations whose performance,
memory traffic, or distributed addressing justify a specialized implementation.
It separates the mathematical operation from the backend that executes it.

This directory is an internal implementation boundary. Public user APIs remain
in their existing simulation, runtime, and algorithm modules. Callers should
reach kernels through those APIs unless they are developing or validating a
kernel.

## Catalog model

The catalog has two layers:

- A **semantic** is a stable, provider-neutral mathematical operation. For
  example, `statevector.apply.matrix_1q.local` describes what is computed.
- An **implementation** is one executable realization of that semantic. It
  records its provider, Python entry point, device, dtype, layout, derivative
  directions, addressing mode, maturity, and fallback behavior.

One semantic may have several implementations. This is intentional: a PyTorch
reference, a direct Triton kernel, and a future FlagTree implementation may all
implement the same contract without creating three meanings for the operation.

The machine-readable records live in [`catalog/`](catalog/). Importing
`flagquantum.kernels.catalog` is metadata-only and must not import Triton,
initialize CUDA, or load an implementation module.

Each implementation also has one **evidence record**. Evidence records link the
implementation ID to repository test nodes, optional checked-in benchmark
artifacts, and the validation lane that must execute device-bound tests. A test
reference is a coverage obligation, not a claim that every pull request ran it.

## Semantic naming

Semantic IDs follow this shape:

```text
<domain>.<operation>.<specialization>.<addressing>
```

Use only components that change the mathematical contract. Provider and tuning
details belong to implementation records. In particular, semantic IDs must not
contain:

- provider names: `triton`, `flagtree`, `cuda`, or `cpu`;
- storage policy such as `complex64`;
- optimization claims such as `fused`, `fast`, or `optimized`;
- lifecycle suffixes such as `v2`;
- application umbrellas such as `qml`.

`qml` is too broad to identify an operator. Quantum machine learning is instead
recorded as the workload tag `quantum_machine_learning`. For example, the
operation used by a variational classifier is
`statevector.apply.rx_rz_sequence.local`; its QML use is discoverable through
metadata without coupling the operator's identity to one application.

Stable catalog IDs such as `FQK-SV-001` are review handles. They do not encode a
version and are never reused for a different semantic.

## Current inventory

The initial catalog describes the code that already exists. It contains 23
semantics and 25 Triton implementation entry points; no planned kernel appears
as an empty machine record.

| Catalog ID | Semantic ID | Implementation symbols |
| --- | --- | --- |
| FQK-SV-001 | `statevector.apply.matrix_1q.local` | `apply_complex64_local_1q`, `single_qubit_matrix` |
| FQK-SV-002 | `statevector.apply.cnot.local` | `apply_complex64_local_cx_inplace` |
| FQK-SV-003 | `statevector.apply.cnot_sequence.local` | `apply_complex64_local_cx_segment`, `cx_sequence` |
| FQK-SV-004 | `statevector.apply.ry_rz_pair.local` | `ry_rz_pair` |
| FQK-SV-005 | `statevector.apply.rx_rz_sequence.local` | `repeated_rx_rz` |
| FQK-SV-006 | `statevector.distributed.transpose_apply_1q` | `apply_complex64_transpose_1q_inplace` |
| FQK-SV-007 | `statevector.transport.control_subspace_pack` | `pack_complex64_control_one` |
| FQK-SV-008 | `statevector.transport.control_subspace_unpack` | `unpack_complex64_control_one` |
| FQK-GR-001 | `gradient.vjp.adjoint_1q.local` | `fused_complex64_local_1q_vjp_adjoint` |
| FQK-GR-002 | `gradient.vjp.reversible_1q.local` | `fused_complex64_local_1q_reversible_vjp` |
| FQK-GR-003 | `gradient.vjp.adjoint_1q.sharded` | `fused_complex64_sharded_1q_vjp_adjoint` |
| FQK-GR-004 | `gradient.jacobian.rx_rz_sequence` | `repeated_rx_rz_tangents` |
| FQK-GR-005 | `gradient.jacobian.pauli_rotation_sequence_2q` | `repeated_rxx_ryy_rzz_tangents` |
| FQK-GR-006 | `gradient.forward_tangent.heisenberg_hva` | `heisenberg_hva_forward_tangents` |
| FQK-MPS-001 | `mps.contract.two_site_gate` | `fused_mps_two_site` |
| FQK-MPS-002 | `mps.contract.two_site_gate_projected` | `fused_mps_range_projection` |
| FQK-MPS-003 | `mps.contract.one_site_gate` | `fused_mps_one_site` |
| FQK-MPS-004 | `mps.environment.transfer_identity_z` | `fused_mps_environment_transfer` |
| FQK-MPS-005 | `mps.environment.transfer_channels` | `fused_mps_environment_channels` |
| FQK-MPS-006 | `mps.gradient.hermitian_observable_adjoint.local` | `fused_mps_hermitian_observable_adjoint` |
| FQK-MPS-007 | `mps.measurement.wire_probabilities.local` | `fused_mps_wire_probabilities` |
| FQK-NUM-001 | `numerics.matmul.complex_batched` | `fused_complex_bmm` |
| FQK-NUM-002 | `numerics.matmul.complex_batched_layout` | `fused_complex_layout_bmm` |

Some current wrappers contain an internal PyTorch path for unsupported devices,
dtypes, or shapes. The implementation record marks this with
`internal_fallback=True`. This describes existing behavior; it does not mean
that a fallback is a second specialized implementation.

## Provider roles

The provider field records execution ownership, not semantic ownership:

- **PyTorch** is the correctness reference and the universal fallback.
- **Triton** is the direct in-repository GPU implementation line. It can evolve
  without waiting for another project and is the primary path for
  quantum-specific fusion and layout work.
- **FlagTree** is a future shared compiler/kernel provider for operations that
  are useful across FlagOS projects and can satisfy the same semantic contract.
- **Torch-FL** may remain an auxiliary integration route, but it is not required
  between FlagQuantum and FlagTree. A direct provider adapter can implement the
  same catalog contract.

Provider selection belongs in dispatch policy outside this catalog. The catalog
describes capabilities and evidence; it does not choose a backend at import
time.

FlagTree also provides a Triton-compatible compiler substitution. Shared source
that runs unchanged under that substitution remains one `triton` implementation
record; it is not duplicated under `kernels/flagtree/` or counted twice in the
inventory. A `flagtree` implementation record is reserved for FlagTree-owned
source, required FlagTree extensions such as TLE, or a materially distinct
support matrix. The real-device compatibility lane in
[`tests/gpu/flagtree/`](../../tests/gpu/flagtree/) verifies distribution
ownership, active CUDA backend identity, dispatch provenance, and all current
shared Triton kernel families under FlagTree.

MPS canonical-transfer absorption is lowered to rank-three batched matrix
multiplication before provider selection. Its current runtime path uses
`torch.bmm`: A800 measurements show that the experimental NUM-001 Triton
implementation is not yet competitive for these shapes. NUM-001 must establish
a repeatable forward and backward win over this baseline before MPS dispatch
selects it. This keeps the mathematical lowering stable while allowing a later
Triton or FlagTree provider change without altering the MPS API.

The MPS-001 two-site gate-contraction route is opt-in through
`FQ_TRITON_MPS_TWO_SITE=1`. The single-pair path authorizes the exact catalog
entry for contiguous CUDA `complex64` tensors outside reverse execution once
the contraction contains at least `2**12` elements for forward-only calls or
`2**18` elements when gradients are required. Equal-shape spatial buckets use
the same catalog route from `2**18` elements. Calls outside those contracts
retain the existing PyTorch contraction, and both routes feed the unchanged
downstream SVD or QR factorization.

The checked-in
[`mps_two_site_dispatch_a800.json`](../../benchmarks/results/local/mps_two_site_dispatch_a800.json)
artifact preserves 30 synchronized groups of 10 invocations for each of four
fixed contraction shapes on `jp-a800-171` and `jp-a800-172`, under stock Triton
3.7.1 and FlagTree 0.7.0. Across all 16 host, compiler, and shape combinations,
the catalog-authorized contraction ranges from `0.81x` to `1.45x` versus eager
PyTorch and from `3.12x` to `4.41x` versus its warm compiled reference for the
forward direction. Forward plus backward ranges from `0.97x` to `1.30x` versus
eager and from `1.84x` to `2.37x` versus warm compiled. Maximum forward
absolute and relative L2 error are `4.20e-9` and `2.95e-7`; maximum gradient
absolute and relative L2 error are `3.49e-7` and `3.33e-7`. The direct wrapper
ranges from `0.68x` to `2.14x` for forward and `0.95x` to `1.21x` for forward
plus backward. Because neither eager comparison wins on every measured shape,
and because this microbenchmark excludes the downstream MPS factorization, the
canonical aggregate records `retain_opt_in`. The warm compiled complex baseline
also carries PyTorch Inductor's warning that complex code generation may be
worse than eager, so it is a compatibility baseline rather than evidence of an
optimized complex kernel. This is bounded development hardware evidence, not a
release gate or scalability claim. Reproduce or validate it with
[`benchmarks/mps_two_site_dispatch.py`](../../benchmarks/mps_two_site_dispatch.py).

MPS-002 projects a gated two-site update into the deterministic fixed-rank QR
range without first materializing the complete two-site matrix. The higher
fixed-rank algorithm remains opt-in through `FQ_MPS_FIXED_RANK_QR=1`, and its
Triton projection is independently opt-in through
`FQ_TRITON_MPS_PROJECTED_TWO_SITE=1`. Eligible calls are contiguous CUDA
`complex64` inference inputs; training retains the differentiable eager
projection because MPS-002 is forward-only. The public fixed-rank path performs
the same QR and reduced contraction after either projection implementation.

The checked-in
[`mps_projected_two_site_dispatch_a800.json`](../../benchmarks/results/local/mps_projected_two_site_dispatch_a800.json)
artifact preserves 30 synchronized groups of 10 invocations for each of four
fixed shapes and ranks on `jp-a800-171` and `jp-a800-172`, under stock Triton
3.7.1 and FlagTree 0.7.0. Across all 16 host, compiler, and shape combinations,
the direct projected kernel ranges from `0.046x` to `1.019x` versus the eager
non-materializing projection and from `0.034x` to `0.459x` versus materialized
PyTorch. The catalog path ranges from `0.046x` to `1.244x` versus eager and
from `0.159x` to `1.885x` versus the warm compiled projection. The complete
opt-in fixed-rank factorization ranges from `0.180x` to `1.288x` versus the
eager factorization, so neither the kernel nor the end-to-end path establishes
an all-shape speed win.

The tradeoff is memory: the kernel's incremental peak allocation is lower in
every measured case, by `5x` to `41x` versus the eager non-materializing
projection and by `12x` versus the materialized baseline. Maximum sample
absolute and relative L2 error are `3.31e-8` and `1.66e-6`; maximum factorized
reconstruction absolute and relative L2 error are `8.77e-8` and `5.15e-6`, and
maximum subspace-projector absolute and relative L2 error are `3.96e-6` and
`9.23e-6`. The canonical aggregate therefore records `retain_opt_in`: this is
a memory-oriented route for constrained workloads, not a default speed path.
Fixed-rank QR itself also remains opt-in because it is approximate and does not
measure discarded weight. The warm compiled complex baseline carries PyTorch
Inductor's warning that complex code generation may be worse than eager. This
is bounded development hardware evidence, not a release gate or scalability
claim. Reproduce or validate it with
[`benchmarks/mps_projected_two_site_dispatch.py`](../../benchmarks/mps_projected_two_site_dispatch.py).

The MPS-003 one-site gate route is enabled by default inside its measured
support window. Set `FQ_TRITON_MPS_ONE_SITE=0` to disable it explicitly. Direct
single-site calls and compiled site buckets authorize the exact catalog entry
for contiguous CUDA `complex64` tensors and two-by-two gates once the flattened
batch-by-bond contraction contains at least `2**12` elements. The custom
autograd boundary preserves tensor and gate gradients; disabled or unsupported
calls retain the existing eager or compiled real/imaginary PyTorch contraction.

The checked-in
[`mps_one_site_dispatch_a800.json`](../../benchmarks/results/local/mps_one_site_dispatch_a800.json)
artifact preserves 30 synchronized groups of 10 invocations for each of five
fixed site-bucket shapes on `jp-a800-171` and `jp-a800-172`, under stock Triton
3.7.1 and FlagTree 0.7.0. Across all 20 host, compiler, and shape combinations,
the complete public dispatch path is `1.09x` to `1.46x` faster than eager
PyTorch and `1.61x` to `2.53x` faster than its warm compiled reference for the
forward direction. Forward plus backward is `1.22x` to `2.80x` faster than
eager and `1.34x` to `2.27x` faster than warm compiled, with maximum forward
absolute error `1.06e-8` and exact gradients for the measured inputs. The
direct wrapper ranges from `0.55x` to `5.25x` for forward and `0.88x` to
`1.13x` for forward plus backward, so the artifact does not claim a direct
microbenchmark win on every shape. The canonical aggregate nevertheless
records `eligible_for_default` because the complete public path wins in both
directions against both public baselines. MPS-003 is therefore the first
`provisional` catalog implementation and uses default dispatch within the
measured support window, with the explicit environment kill switch above. The
result is bounded development hardware evidence, not a release gate or
scalability claim. Reproduce or validate it with
[`benchmarks/mps_one_site_dispatch.py`](../../benchmarks/mps_one_site_dispatch.py).

The MPS-004 identity/Pauli-Z environment-transfer route is enabled by default
inside its measured support window. Set `FQ_TRITON_MPS_ENVIRONMENT=0` to disable
it explicitly. Dispatch authorizes the exact catalog entry before importing
Triton and selects it only for contiguous CUDA `complex64` forward inputs
without gradients, bond dimensions at most 32, and contraction work at most
`2**22`. Other inputs remain on the existing PyTorch eager or compiled path,
and route counts are exposed through `site_kernel_stats()`.

The checked-in
[`mps_environment_dispatch_a800.json`](../../benchmarks/results/local/mps_environment_dispatch_a800.json)
artifact preserves 30 synchronized groups of 10 invocations for each Identity
and Pauli-Z case on `jp-a800-171` and `jp-a800-172`, under stock Triton 3.7.1
and FlagTree 0.7.0. Across the fixed eight-case support matrix, the direct
kernel wrapper is `1.39x` to `8.90x` faster than the equivalent PyTorch einsum.
The complete public dispatch path is `3.08x` to `10.60x` faster than the eager
reference and `2.09x` to `7.09x` faster than its warm compiled reference, with
maximum absolute error `2.53e-8`. The canonical aggregate records
`eligible_for_default`, and MPS-004 is now a `provisional` implementation with
default dispatch inside the measured window and the explicit kill switch above.
The result is bounded development hardware evidence, not a release gate or
scalability claim. Reproduce or validate it with
[`benchmarks/mps_environment_dispatch.py`](../../benchmarks/mps_environment_dispatch.py).

MPS-005 groups as many as eight observable channels in each Triton program so
the channels reuse site-tensor loads. Its implementation supports contiguous
CUDA `complex64` forward inputs without gradients, at most 32 channels, bond
dimensions at most 16, and contraction work at most `2**23`. The wrapper keeps
the exact PyTorch contraction as its explicit fallback outside that measured
window. The same `FQ_TRITON_MPS_ENVIRONMENT=1` rollout switch routes eligible
multi-channel transfers through the exact MPS-005 catalog entry and reports
them separately through `site_kernel_stats()`.

The checked-in
[`mps_environment_channels_dispatch_a800.json`](../../benchmarks/results/local/mps_environment_channels_dispatch_a800.json)
artifact preserves 30 synchronized groups of 10 invocations for each of five
fixed channel-transfer shapes on `jp-a800-171` and `jp-a800-172`, under stock
Triton 3.7.1 and FlagTree 0.7.0. Across all 20 host, compiler, and shape
combinations, the complete public dispatch path is `1.03x` to `4.21x` faster
than eager PyTorch and `3.69x` to `8.48x` faster than its warm compiled
reference, with maximum absolute error `3.22e-8`. The direct kernel wrapper is
`0.79x` to `6.76x` the speed of the direct einsum: FlagTree 0.7.0 loses that
microbenchmark at the largest `16 x 8 x 16 x 16` boundary shape on both hosts,
while the public dispatch path still wins against both public baselines. The
canonical aggregate therefore records `eligible_for_default`; this authorizes
a separate dispatch-promotion review but does not itself change the opt-in
policy. The result is bounded development hardware evidence, not a release
gate or scalability claim. Reproduce or validate it with
[`benchmarks/mps_environment_channels_dispatch.py`](../../benchmarks/mps_environment_channels_dispatch.py).

MPS-006 evaluates the local tensor VJP of a Hermitian observable contribution
without constructing a per-site autograd graph. Hermitian left and right
environments and a Hermitian two-by-two operator make the two Wirtinger terms
equal, so the implementation computes one contraction and scales it by two.
The Triton path supports contiguous CUDA `complex64` inputs, bond dimensions at
most 64, and at most `2**25` scalar contraction work; its wrapper uses the same
closed-form PyTorch contraction outside that measured window. Runtime dispatch
is opt-in through `FQ_TRITON_MPS_OBSERVABLE_ADJOINT=1` and additionally requires
the caller to establish the Hermitian semantic invariant. Other inputs retain
the existing per-site autograd path. Route and fallback counts are exposed by
`site_kernel_stats()`, while catalog-route events report the exact semantic and
implementation IDs plus the active Triton or FlagTree compiler provenance.
The checked-in
[`mps_observable_adjoint_dispatch_a800.json`](../../benchmarks/results/local/mps_observable_adjoint_dispatch_a800.json)
artifact preserves all 30 synchronized samples per case, peak memory, accuracy,
compiler identity, and environment metadata from `jp-a800-171` and
`jp-a800-172`. Its fixed four-shape matrix records a `5.20x` to `24.84x`
speedup over the same PyTorch autograd semantic with stock Triton 3.7.1 and a
`3.13x` to `9.97x` speedup with FlagTree 0.7.0. This is bounded development
hardware evidence, not a release gate or scalability claim. Reproduce or
validate it with
[`benchmarks/mps_observable_adjoint_dispatch.py`](../../benchmarks/mps_observable_adjoint_dispatch.py).

MPS-007 fuses the complex magnitude, left/right bond reduction, and
normalization needed to obtain the two physical-index probabilities at one MPS
site. Its implementation supports contiguous CUDA `complex64` tensors without
gradients and at most `2**12` left-by-right bond elements; the wrapper retains
the exact PyTorch reduction for other inputs. Runtime dispatch is opt-in through
`FQ_TRITON_MPS_WIRE_PROBABILITIES=1`; eligible calls from the public MPS
sampling path authorize the exact MPS-007 catalog entry before importing
Triton. Route and fallback counts are exposed through `site_kernel_stats()`,
and catalog-route events include the active stock Triton or FlagTree compiler
provenance. The switch remains an integration-validation route rather than a
default performance selection: synchronized public-path measurements must show
an end-to-end win before default dispatch is considered.

The checked-in
[`mps_wire_probability_dispatch_a800.json`](../../benchmarks/results/local/mps_wire_probability_dispatch_a800.json)
artifact preserves 30 synchronized groups of 100 invocations for each case on
`jp-a800-171` and `jp-a800-172`, under stock Triton 3.7.1 and FlagTree 0.7.0.
Across the fixed sequential-sampling shape matrix, the direct kernel wrapper is
`2.22x` to `3.40x` faster than the equivalent PyTorch reduction, with maximum
absolute error `8.94e-8`. The complete public dispatch path is instead `4.8%`
to `9.6%` slower because its per-wire safety checks synchronize the device.
The canonical aggregate therefore records `retain_opt_in`; this is bounded
development-hardware evidence, not a release gate or scalability claim.
Reproduce or validate it with
[`benchmarks/mps_wire_probability_dispatch.py`](../../benchmarks/mps_wire_probability_dispatch.py).

## Capability matching

`catalog.match_kernel_implementations` answers which cataloged implementations
declare support for a request. A request identifies the semantic, device,
dtype, layout, derivative direction, addressing requirements, and optional
provider and maturity filters.

```python
from flagquantum.kernels.catalog import (
    KernelRequest,
    match_kernel_implementations,
)

result = match_kernel_implementations(
    KernelRequest(
        semantic_id="statevector.apply.matrix_1q.local",
        device="cuda",
        dtype="complex64",
        layout="flat_statevector",
        direction="backward",
        addressing=("local",),
    )
)
```

The matcher returns every compatible, evidenced implementation in catalog
order, plus structured reasons for each rejected implementation. It does not:

- rank Triton, FlagTree, or PyTorch providers;
- import implementation modules or probe whether optional packages are loaded;
- inspect live hardware or initialize CUDA;
- execute an internal fallback;
- turn a declared capability into observed execution evidence.

An internal PyTorch fallback therefore does not make a Triton implementation a
CPU specialization. A CPU request rejects an implementation whose cataloged
device is CUDA. Runtime policy may use matcher output as one input to provider
selection, but loading, live availability checks, priorities, and fallback
execution remain separate responsibilities.

## Target inventory: 100 semantics and 800 implementations

The program target is **100 semantic families** represented by approximately
**800 implementation records**. An implementation is a verified combination of
provider, device, dtype, layout, addressing mode, and derivative direction. It
is not a promise of 800 unrelated handwritten source files.

### Foundation layer: 70 semantics / 560 implementations

| Domain | Semantics | Implementations | Typical operations | Application coverage |
| --- | ---: | ---: | --- | --- |
| Statevector evolution | 18 | 144 | dense and diagonal gates, controlled gates, Pauli rotations, gate sequences | circuit simulation, QFT, QPE, Grover, variational circuits |
| Tensor networks | 12 | 96 | MPS/MPO contraction, canonicalization, truncation, environment updates | MPS state preparation, DMRG, many-body simulation |
| Differentiation | 10 | 80 | adjoint VJP, reversible VJP, JVP, parameter-shift reduction, tangent propagation | VQE, QAOA, VQC, VQLS, QCBM, quantum Fisher information |
| Complex numerics | 10 | 80 | batched matmul, reductions, normalization, eigensolver/SVD support | statevector and tensor-network backends, chemistry |
| Measurement and sampling | 8 | 64 | probability reduction, marginal sampling, expectation and variance | circuit sampling, shadows, mitigation, observable estimation |
| Distributed data movement | 7 | 56 | pack/unpack, transpose, shard exchange, collective-ready reductions | multi-GPU statevector and MPS training |
| Representation transforms | 5 | 40 | basis permutation, bit indexing, sparse/dense and Pauli transforms | compilation, Hamiltonian simulation, interoperability |
| **Foundation total** | **70** | **560** |  |  |

### Application layer: 30 semantics / 240 implementations

Application-layer kernels fuse recurring structures only after a foundation
implementation and an end-to-end workload establish the contract.

| Application family | Semantics | Implementations | Candidate fused structures | Workloads served |
| --- | ---: | ---: | --- | --- |
| Algorithm primitives and search | 5 | 40 | QFT stages, phase kickback, reflection and amplitude amplification | QFT, QPE, Hadamard test, Grover, amplitude estimation, Shor, Simon |
| Linear systems and Hamiltonian simulation | 5 | 40 | block encoding, LCU select/prepare, Trotter layers, qDrift batches | HHL, VQLS, QSVT, QSP, LCU, Trotter-Suzuki, qDrift |
| Variational computing and QML | 5 | 40 | hardware-efficient layers, QAOA mixers, covariance/Fisher reductions | VQE, VQD, QAOA, VQC, QCBM, CVQNN |
| State preparation and many-body | 4 | 32 | Möttönen stages, isometry blocks, MPS sweeps, lattice layers | state preparation, DMRG, Fermi-Hubbard and Ising models |
| Chemistry and optimization | 3 | 24 | Pauli-term batching, fermionic excitation blocks, objective reduction | molecular VQE, SQD, QSVM, QUBO and constrained optimization |
| Error mitigation and correction | 3 | 24 | noisy-shot aggregation, syndrome extraction, decoder primitives | ZNE, REM, PEC, CDR, surface code, qLDPC, BP+LSD |
| Open systems and control | 3 | 24 | trajectory batches, Lindblad updates, pulse propagation | TJM, driven-system gradients, non-Markovian dynamics, pulse control |
| Photonics and quantum networks | 2 | 16 | interferometer transforms, Fock reductions, link-event batches | linear optics, photonic learning, link loss and entanglement protocols |
| **Application total** | **30** | **240** |  |  |

These counts are portfolio envelopes. A candidate enters the machine catalog
only when it has an executable symbol, a reference contract, and validation
evidence. Until then it remains a planning item in this README.

## Status and maturity

Planning and implementation use separate status vocabularies:

- **candidate**: a workload trace suggests a useful semantic, but its contract
  or value is not yet established;
- **planned**: the contract, reference, owner, and benchmark shape are agreed;
- **implemented**: at least one executable implementation is cataloged and
  passes required validation.

Implementation maturity is independent:

- **experimental**: correct on a bounded support matrix; API and tuning may
  change;
- **provisional**: dispatch and support matrix are exercised by integration and
  performance tests;
- **stable**: compatibility, fallback, accuracy, and performance regression
  policies are maintained.

The current 23 semantics and 25 implementations are implemented. MPS-003 and
MPS-004 are provisional after their evidenced default-dispatch promotions; the
other 23 implementations remain experimental. The rest of the 100/800 portfolio
is planned or candidate work, not shipped capability.

## Validation contract

Every new implementation must provide evidence in four dimensions:

1. **Correctness** against the PyTorch reference over representative shapes,
   parameters, addressing cases, and deterministic seeds.
2. **Gradient correctness** for every declared backward, VJP, JVP, or Jacobian
   direction, including complex-number conventions.
3. **Capability behavior** for unsupported devices, dtypes, shapes, layouts,
   and optional dependencies. Fallback or refusal must be explicit.
4. **Performance evidence** after correctness is established: warm-up policy,
   synchronization, compiler identity, shape matrix, memory use, and baseline
   must be recorded.

GPU validation can use `jp-a800-171` and `jp-a800-172`. Cross-machine evidence
must record GPU, CUDA, PyTorch, Triton or FlagTree identity, commit, command,
shape matrix, warm-up, repetitions, and raw results. A speedup is meaningful
only for the same semantic, dtype, layout, accuracy tolerance, and synchronization
policy.

Catalog validation is intentionally CPU-only and dependency-light. It checks
identity, references, naming, and metadata without importing an accelerator
provider.

FlagTree wheel validation must use the pinned container lane documented in
[`tests/gpu/flagtree/`](../../tests/gpu/flagtree/). The A800 hosts currently
provide GLIBC 2.35, while the published NVIDIA FlagTree wheels require GLIBC
2.38 or newer. A host import failure at that boundary is an environment
incompatibility, not kernel evidence and not authorization to relabel stock
Triton as FlagTree.

The evidence catalog enforces these minimums:

- every implementation has correctness evidence;
- every implementation that declares backward, VJP, JVP, or Jacobian support
  has gradient evidence;
- every wrapper with an internal fallback has capability/fallback evidence;
- every referenced test node and checked-in artifact exists;
- device tests name an explicit required lane, currently `gpu_scheduled` for
  Triton implementations.

The ordinary PR lanes validate the catalog and its references. An executed GPU
lane, or a recorded run on `jp-a800-171` or `jp-a800-172`, is required before a
device result is described as observed hardware evidence.

## Adding a kernel

1. Identify a repeated workload bottleneck and state the provider-neutral
   mathematical contract.
2. Reuse an existing semantic if the inputs, outputs, addressing, and numerical
   meaning match. Add a semantic only when the contract is genuinely new.
3. Add the executable implementation under the appropriate provider directory.
4. Register its support matrix in `catalog/implementations.py`; never encode
   provider, dtype, tuning, or maturity in the semantic ID.
5. Add correctness, derivative, fallback, and capability tests.
6. Run catalog validation and provider-specific tests.
7. Add benchmark evidence only after correctness gates pass.
8. Connect runtime dispatch in a separate, reviewable change when possible.

A kernel PR should normally cover one semantic family or one coherent provider
slice. It should not combine catalog design, broad runtime dispatch changes, and
unrelated numerical refactors.

## Near-term sequence

The next development sequence is:

1. keep this catalog synchronized with the existing Triton entry points;
2. attach each implementation to its correctness and benchmark evidence;
3. introduce capability-based dispatch without changing user-facing APIs;
4. add direct FlagTree implementations for semantics with cross-project value;
5. expand foundation semantics from measured workload traces;
6. promote recurring application structures only after end-to-end evidence.

This keeps the direct FlagQuantum kernel line primary and independently usable,
while FlagTree and Torch-FL remain compatible provider and integration paths
rather than mandatory ownership layers.
