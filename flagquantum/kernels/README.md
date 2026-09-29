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

The initial catalog describes the code that already exists. It contains 18
semantics and 20 Triton implementation entry points; no planned kernel appears
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

The current 18 semantics and 20 implementations are implemented and
experimental. The rest of the 100/800 portfolio is planned or candidate work,
not shipped capability.

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
