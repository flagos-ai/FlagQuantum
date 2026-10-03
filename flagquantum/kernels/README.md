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

The catalog describes the code that already exists. It contains 26 semantics,
28 Triton implementation entry points, and five FlagTree TLE implementation
entry points; no planned kernel appears as an empty machine record.

| Catalog ID | Semantic ID | Implementation symbols |
| --- | --- | --- |
| FQK-SV-001 | `statevector.apply.matrix_1q.local` | `apply_complex64_local_1q`, `single_qubit_matrix`, `apply_complex64_local_1q_tle` (FlagTree TLE) |
| FQK-SV-002 | `statevector.apply.cnot.local` | `apply_complex64_local_cx_inplace` |
| FQK-SV-003 | `statevector.apply.cnot_sequence.local` | `apply_complex64_local_cx_segment`, `cx_sequence` |
| FQK-SV-004 | `statevector.apply.ry_rz_pair.local` | `ry_rz_pair` |
| FQK-SV-005 | `statevector.apply.rx_rz_sequence.local` | `repeated_rx_rz` |
| FQK-SV-006 | `statevector.distributed.transpose_apply_1q` | `apply_complex64_transpose_1q_inplace`, `apply_complex64_transpose_1q_tle_inplace` (FlagTree TLE) |
| FQK-SV-007 | `statevector.transport.control_subspace_pack` | `pack_complex64_control_one`, `pack_complex64_control_one_tle` (FlagTree TLE) |
| FQK-SV-008 | `statevector.transport.control_subspace_unpack` | `unpack_complex64_control_one`, `unpack_complex64_control_one_tle` (FlagTree TLE) |
| FQK-GR-001 | `gradient.vjp.adjoint_1q.local` | `fused_complex64_local_1q_vjp_adjoint` |
| FQK-GR-002 | `gradient.vjp.reversible_1q.local` | `fused_complex64_local_1q_reversible_vjp` |
| FQK-GR-003 | `gradient.vjp.adjoint_1q.sharded` | `fused_complex64_sharded_1q_vjp_adjoint`, `fused_complex64_sharded_1q_vjp_adjoint_tle` (FlagTree TLE) |
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
| FQK-MEAS-001 | `measurement.probabilities.statevector` | `statevector_probabilities` |
| FQK-MEAS-002 | `measurement.expectation.pauli_product.statevector` | `statevector_pauli_expectation` |
| FQK-MEAS-003 | `measurement.probabilities.marginal.statevector` | `statevector_marginal_probabilities` |
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
- **FlagTree** is the shared compiler/kernel provider for operations that are
  useful across FlagOS projects and can satisfy the same semantic contract.
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

`FQKI-FLAGTREE-SV-001-A` is the first provider-owned FlagTree record. It reuses
the `statevector.apply.matrix_1q.local` semantic but requires FlagTree 0.7.0's
TLE `load` primitive. Vector state loads use TLE async annotations; the eight
matrix scalars deliberately use ordinary `tl.load`, because FlagTree 0.7.0's
NVIDIA lowering crashes when a scalar load is marked async. The wrapper verifies
that the installed `flagtree` distribution owns the `triton` namespace and maps
the CUDA runtime target to FlagTree's `nvidia` TLE whitelist before importing or
launching the extension. It is an explicit internal entry point and is not
selected by the default runtime.

The checked-in
[`flagtree_tle_local_1q_a800.json`](../../benchmarks/results/local/flagtree_tle_local_1q_a800.json)
artifact records 30 synchronized groups of 10 invocations for four fixed
complex64 state sizes on `jp-a800-171` and `jp-a800-172` with FlagTree 0.7.0.
The TLE wrapper reaches `0.866x` to `0.942x` the speed of the same gate using
shared Triton source, while reaching `1.016x` to `11.536x` the speed of the
PyTorch reference. Maximum absolute and relative L2 errors are `9.84e-7` and
`3.92e-8`. The canonical decision is `retain_explicit`: this proves a direct
FlagTree-owned provider slice and its fail-closed capability boundary, but does
not authorize default dispatch. It is bounded single-device development
evidence, not a release gate or scalability claim. Reproduce or validate it
with
[`benchmarks/flagtree_tle_local_1q.py`](../../benchmarks/flagtree_tle_local_1q.py).

The same two-host artifact also establishes the dispatch window for
`FQKI-TRITON-SV-001-A`, the shared Triton forward implementation used by the
distributed statevector executor. Without an override, contiguous CUDA
`complex64` shards select it only for batch-one shapes with `2**10`, `2**16`,
`2**20`, or `2**24` amplitudes. Set `FQ_STATEVECTOR_TRITON_LOCAL_1Q=0` to use
the PyTorch route, or set it to `1` to opt into the catalog implementation for
other supported contiguous shapes. Across both A800 hosts, shared Triton is
`1.160x` to `12.247x` faster than the same-semantic PyTorch reference and wins
every measured case. The catalog route and its kill switch are covered by
runtime integration tests, so SV-001-A is `provisional` within this exact
default window. This is bounded single-device development evidence, not a
multi-rank scalability or release claim.

`FQKI-FLAGTREE-SV-006-A` applies the same explicit provider boundary to the
fused distributed transpose and one-qubit gate. TLE async annotations cover
both the retained local half-shard and the received remote half-shard; matrix
scalars retain ordinary `tl.load` for the FlagTree 0.7.0 limitation above. The
wrapper validates the exact contiguous `[B, 2**n]` state and `[B, 2**(n-1)]`
received layouts before probing FlagTree, and default distributed dispatch
continues to select the shared Triton implementation.

The checked-in
[`flagtree_tle_transpose_1q_a800.json`](../../benchmarks/results/local/flagtree_tle_transpose_1q_a800.json)
artifact records 30 synchronized groups of 10 invocations for both exchanged
bit values and four fixed state sizes from `2**10` through `2**24` amplitudes
on `jp-a800-171` and `jp-a800-172` with FlagTree 0.7.0. TLE reaches `0.875x`
to `0.968x` the speed of the shared Triton implementation and `1.472x` to
`12.499x` the speed of the PyTorch reference. Maximum absolute and relative L2
errors are `4.81e-7` and `3.70e-8`. The canonical decision is
`retain_explicit`: the provider is correct and directly measurable but lacks a
cross-matrix performance win, so it does not authorize default dispatch. This
is single-device development evidence, not distributed scalability or release
evidence. Reproduce or validate it with
[`benchmarks/flagtree_tle_transpose_1q.py`](../../benchmarks/flagtree_tle_transpose_1q.py).

`FQKI-FLAGTREE-SV-007-A` and `FQKI-FLAGTREE-SV-008-A` extend the same explicit
provider boundary to distributed-CX control-subspace pack and unpack. Their TLE
source uses async loads for the non-contiguous state gather and packed-buffer
read while preserving the established `flat_statevector` ↔ `packed_subspace`
semantic contract. The wrappers require contiguous CUDA `complex64` tensors,
validate the address range before probing FlagTree, and remain outside default
dispatch.

The checked-in
[`flagtree_tle_control_transport_a800.json`](../../benchmarks/results/local/flagtree_tle_control_transport_a800.json)
artifact records the same 30 synchronized groups of 10 invocations for pack and
unpack across four fixed state sizes from `2**10` through `2**24` amplitudes on
`jp-a800-171` and `jp-a800-172` with FlagTree 0.7.0. Against the existing shared
Triton source, TLE reaches `0.867x` to `0.921x` for pack and `0.828x` to
`0.972x` for unpack; all measured values are exact against the indexed
reference. The canonical decision is therefore `retain_explicit`: this
establishes a direct FlagTree transport provider and a reproducible optimization
surface, but does not authorize default dispatch. This is single-device
development evidence only, not a distributed scalability or release claim.
Reproduce or validate it with
[`benchmarks/flagtree_tle_control_transport.py`](../../benchmarks/flagtree_tle_control_transport.py).

`FQKI-FLAGTREE-GR-003-A` extends the explicit provider boundary to one rank's
sharded one-qubit adjoint VJP. The four complex local and remote ket/adjoint
streams use TLE async loads, while the selected matrix and derivative scalars
remain ordinary `tl.load` operations for the FlagTree 0.7.0 scalar-lowering
limitation. The wrapper validates matching nonempty contiguous CUDA `complex64`
chunks and a binary rank basis before probing FlagTree. Default distributed
reverse execution continues to select the shared Triton implementation.

The checked-in
[`flagtree_tle_sharded_adjoint_a800.json`](../../benchmarks/results/local/flagtree_tle_sharded_adjoint_a800.json)
artifact records 30 synchronized groups of 10 invocations for both rank-basis
values and four fixed chunk sizes from `2**10` through `2**24` elements on
`jp-a800-171` and `jp-a800-172` with FlagTree 0.7.0. TLE reaches `0.903x` to
`0.952x` the speed of shared Triton and `1.708x` to `17.375x` the speed of the
PyTorch reference. Maximum adjoint absolute and relative L2 errors are
`4.77e-7` and `4.08e-8`; maximum gradient absolute and relative errors are
`3.66e-4` and `7.42e-7`. The canonical decision is `retain_explicit`: the
provider establishes a directly measurable FlagTree gradient slice but lacks a
cross-matrix win, so it does not authorize default dispatch. This rank-local,
single-device result is development evidence, not distributed scalability or
release evidence. Reproduce or validate it with
[`benchmarks/flagtree_tle_sharded_adjoint.py`](../../benchmarks/flagtree_tle_sharded_adjoint.py).

MPS canonical-transfer absorption is lowered to rank-three batched matrix
multiplication before provider selection. Its current runtime path uses
`torch.bmm`: A800 measurements show that the experimental NUM-001 Triton
implementation is not yet competitive for these shapes. NUM-001 must establish
a repeatable forward and backward win over this baseline before MPS dispatch
selects it. This keeps the mathematical lowering stable while allowing a later
Triton or FlagTree provider change without altering the MPS API.

`FQKI-TRITON-MEAS-001-A` computes the full flat-statevector probability tensor
and its first-order complex gradient. Runtime dispatch is enabled by default
only for contiguous CUDA `complex64` statevectors with shape `(1, 2**24)`, no
conjugate or negative view bits, and the complete canonical wire order. Set
`FQ_TRITON_STATEVECTOR_PROBABILITIES=0` to use the PyTorch reference path
explicitly. All other shapes, layouts, devices, dtypes, and wire orders return
to the existing reference path before importing Triton. The checked-in
[`statevector_probability_kernel_a800.json`](../../benchmarks/results/local/statevector_probability_kernel_a800.json)
artifact records 30 synchronized groups of 10 invocations for five fixed
complex64 shapes from 1,024 through 16,777,216 amplitudes on `jp-a800-171` and
`jp-a800-172` with stock Triton 3.7.1. Maximum probability and gradient absolute
errors are `9.32e-10` and `3.34e-8`. Triton reaches `0.564x` to `2.999x` the
PyTorch forward speed and `0.886x` to `3.078x` the PyTorch forward/backward
speed. The approximately `3x` win is confined to the 16,777,216-amplitude case;
smaller cases remain at parity or slower, so the canonical decision is
`retain_experimental`; this direct evidence alone does not authorize broader
dispatch. Reproduce or validate the direct kernel evidence with
[`benchmarks/statevector_probability_kernel.py`](../../benchmarks/statevector_probability_kernel.py).

The checked-in
[`statevector_probability_dispatch_a800.json`](../../benchmarks/results/local/statevector_probability_dispatch_a800.json)
artifact measures the complete public full-probability path for the exact
default window on `jp-a800-171` and `jp-a800-172`, under stock Triton 3.7.1 and
FlagTree 0.7.0. Across all four host/compiler runs, public forward dispatch is
`1.732x` to `1.742x` faster than the identical public path with the kernel
disabled, and forward plus backward is `1.586x` to `1.605x` faster. Maximum
probability and gradient absolute errors are `1.14e-13` and `5.21e-10`; maximum
relative L2 errors are `6.65e-8` and `4.82e-8`. The runner rejects any host,
compiler, or direction below its `1.0x` floor. The canonical aggregate records
`eligible_for_default`, so MEAS-001 is now a `provisional` implementation with
default dispatch inside this exact measured window and the explicit kill switch
above. This is bounded single-device development evidence against the identical
public PyTorch path, not a release gate, framework-wide comparison, or
scalability claim. Reproduce or validate it with
[`benchmarks/statevector_probability_dispatch.py`](../../benchmarks/statevector_probability_dispatch.py).

`FQKI-TRITON-MEAS-002-A` evaluates exact Pauli-product expectations directly
from a batched flat statevector. It fuses basis-index permutation, X/Y/Z phase,
and complex inner-product work, and its explicit backward computes the complex
gradient of the real expectation. Wire zero addresses the most-significant
statevector bit, matching the simulation contract. Runtime dispatch is enabled
by default only for contiguous CUDA `complex64` statevectors. The dispatched
shape is exactly `(1, 2**24)`, the state must have no conjugate or negative view
bits, and the Pauli product must be exactly `X0 * Y12 * Z23`. Set
`FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION=0` to use the PyTorch reference path
explicitly. All other shapes, layouts, devices, dtypes, and Pauli products
return to the existing reference path before importing Triton. The checked-in
[`statevector_pauli_expectation_kernel_a800.json`](../../benchmarks/results/local/statevector_pauli_expectation_kernel_a800.json)
artifact records 30 synchronized groups of 10 invocations for five fixed
three-factor Pauli products over complex64 shapes from 1,024 through 16,777,216
amplitudes on `jp-a800-171` and `jp-a800-172` with stock Triton 3.7.1. Maximum
expectation absolute and relative L2 errors are `5.59e-9` and `2.75e-7`; the
explicit gradient matches exactly for this matrix. Against FlagQuantum's
current sequential PyTorch Pauli-product reference, Triton reaches `5.239x` to
`36.581x` the forward speed and `4.627x` to `37.859x` the forward/backward
speed. The evidence covers one operator pattern and two development hosts, so
the canonical decision remains `retain_experimental`; this direct evidence
alone does not authorize broader dispatch. Reproduce or validate the direct
kernel evidence with
[`benchmarks/statevector_pauli_expectation_kernel.py`](../../benchmarks/statevector_pauli_expectation_kernel.py).

The checked-in
[`statevector_pauli_expectation_dispatch_a800.json`](../../benchmarks/results/local/statevector_pauli_expectation_dispatch_a800.json)
artifact measures the complete public `Circuit.expectation_ps` path for the
exact default window on `jp-a800-171` and `jp-a800-172`, under stock Triton
3.7.1 and FlagTree 0.7.0. Across all four host/compiler runs, public forward
dispatch is `39.313x` to `40.997x` faster than the identical public path with
the kernel disabled, and forward plus backward is `41.350x` to `54.467x`
faster. Maximum expectation absolute and relative L2 errors are `5.83e-11` and
`1.58e-7`; the explicit gradient matches exactly. The runner rejects any host,
compiler, or direction below its `1.0x` floor. The canonical aggregate records
`eligible_for_default`, so MEAS-002 is now a `provisional` implementation with
default dispatch inside this exact measured window and the explicit kill switch
above. This is bounded single-device development evidence against the identical
public PyTorch path, not a release gate, framework-wide comparison, or
scalability claim. Reproduce or validate it with
[`benchmarks/statevector_pauli_expectation_dispatch.py`](../../benchmarks/statevector_pauli_expectation_dispatch.py).

`FQKI-TRITON-MEAS-003-A` computes a joint marginal distribution directly from
a batched flat statevector without materializing the full probability tensor.
Wire zero addresses the most-significant statevector bit, and output bits retain
the exact order requested by the caller. The direct Triton path accepts
contiguous CUDA `complex64` statevectors with at most 30 wires and selections of
at most eight wires. The measured Triton window requires at least `2**16`
amplitudes per statevector and `2**20` amplitudes across the batch; other valid
inputs retain an exact differentiable PyTorch fallback. Runtime dispatch is
enabled by default only for a contiguous CUDA `complex64` statevector with
shape `(1, 2**24)` and exactly four selected wires. Set
`FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES=0` to use the PyTorch reference
path explicitly. All other shapes, devices, dtypes, and selections return to
the existing reference path before importing Triton. The checked-in
[`statevector_marginal_probability_kernel_a800.json`](../../benchmarks/results/local/statevector_marginal_probability_kernel_a800.json)
artifact records 30 synchronized groups of 10 invocations for five fixed
complex64 workloads from 1,048,576 through 16,777,216 total amplitudes on
`jp-a800-171` and `jp-a800-172` with stock Triton 3.7.1. Maximum probability
and gradient absolute errors are `1.49e-8` and `5.27e-9`. Against the exact
same-semantic eager PyTorch reference, Triton reaches `1.080x` to `2.724x` the
forward speed and `1.014x` to `3.625x` the forward/backward speed; every fixed
host/workload result must remain at or above the runner's `1.0x` performance
floor. The evidence covers one dtype and two development hosts, so the
canonical decision remains `retain_experimental`; it does not authorize
broader default dispatch or a release claim. Reproduce or validate the direct
kernel evidence with
[`benchmarks/statevector_marginal_probability_kernel.py`](../../benchmarks/statevector_marginal_probability_kernel.py).

The checked-in
[`statevector_marginal_probability_dispatch_a800.json`](../../benchmarks/results/local/statevector_marginal_probability_dispatch_a800.json)
artifact measures the complete public marginal-probability path for the exact
default window on `jp-a800-171` and `jp-a800-172`, under stock Triton 3.7.1 and
FlagTree 0.7.0. Across all four host/compiler runs, public forward dispatch is
`1.93x` to `2.17x` faster than the identical public path with the kernel
disabled, and forward plus backward is `1.67x` to `2.05x` faster. Maximum
probability and gradient absolute errors are `7.45e-9` and `2.61e-10`. The
runner rejects any host, compiler, or direction below its `1.0x` floor. The
canonical aggregate records `eligible_for_default`, so MEAS-003 is now a
`provisional` implementation with default dispatch inside this exact measured
window and the explicit kill switch above. This is bounded single-device
development evidence, not a release gate, framework-wide comparison, or
scalability claim. Reproduce or validate it with
[`benchmarks/statevector_marginal_probability_dispatch.py`](../../benchmarks/statevector_marginal_probability_dispatch.py).

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
the channels reuse site-tensor loads. The route is enabled by default inside
its measured support window and shares the explicit
`FQ_TRITON_MPS_ENVIRONMENT=0` kill switch with MPS-004. Its implementation
supports contiguous CUDA `complex64` forward inputs without gradients, at most
32 channels, bond dimensions at most 16, and contraction work at most `2**23`.
The wrapper keeps the exact PyTorch contraction as its explicit fallback
outside that measured window. Eligible multi-channel transfers authorize the
exact MPS-005 catalog entry and are reported separately through
`site_kernel_stats()`.

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
canonical aggregate records `eligible_for_default`, and MPS-005 is now a
`provisional` implementation with default dispatch inside the measured window
and the explicit shared kill switch above. The result is bounded development
hardware evidence, not a release gate or scalability claim. Reproduce or
validate it with
[`benchmarks/mps_environment_channels_dispatch.py`](../../benchmarks/mps_environment_channels_dispatch.py).

MPS-006 evaluates the local tensor VJP of a Hermitian observable contribution
without constructing a per-site autograd graph. Hermitian left and right
environments and a Hermitian two-by-two operator make the two Wirtinger terms
equal, so the implementation computes one contraction and scales it by two.
The Triton path supports contiguous CUDA `complex64` inputs, bond dimensions at
most 64, and at most `2**25` scalar contraction work; its wrapper uses the same
closed-form PyTorch contraction outside that measured window. Runtime dispatch
is enabled by default inside that window and additionally requires the caller
to establish the Hermitian semantic invariant. Set
`FQ_TRITON_MPS_OBSERVABLE_ADJOINT=0` to use the reference path explicitly.
Other inputs retain the existing per-site autograd path. Route and fallback
counts are exposed by `site_kernel_stats()`, while catalog-route events report
the exact semantic and implementation IDs plus the active Triton or FlagTree
compiler provenance.
The checked-in
[`mps_observable_adjoint_dispatch_a800.json`](../../benchmarks/results/local/mps_observable_adjoint_dispatch_a800.json)
artifact preserves all 30 synchronized samples per case, peak memory, accuracy,
compiler identity, and environment metadata from `jp-a800-171` and
`jp-a800-172`. Its fixed four-shape matrix records a `5.20x` to `24.84x`
speedup over the same PyTorch autograd semantic with stock Triton 3.7.1 and a
`3.13x` to `9.97x` speedup with FlagTree 0.7.0. The canonical aggregate records
`eligible_for_default`, and MPS-006 is now a `provisional` implementation with
default dispatch inside the measured window and the explicit kill switch
above. This is bounded development hardware evidence, not a release gate or
scalability claim. Reproduce or validate it with
[`benchmarks/mps_observable_adjoint_dispatch.py`](../../benchmarks/mps_observable_adjoint_dispatch.py).

MPS-007 fuses the complex magnitude, left/right bond reduction, and
normalization needed to obtain the two physical-index probabilities at one MPS
site. Its implementation supports contiguous CUDA `complex64` tensors without
gradients and at most `2**12` left-by-right bond elements; the wrapper retains
the exact PyTorch reduction for other inputs. Runtime dispatch is enabled by
default inside that measured window. Set
`FQ_TRITON_MPS_WIRE_PROBABILITIES=0` to use the reference path explicitly.
Eligible calls from the public MPS sampling path authorize the exact MPS-007
catalog entry before importing Triton. Route and fallback counts are exposed
through `site_kernel_stats()`, and the first catalog-route event in each
statistics window includes the active stock Triton or FlagTree compiler
provenance. The Triton result is already normalized, so the public path returns
it directly and enqueues finite-value validation with `torch._assert_async`
instead of synchronizing the host for every wire.

The checked-in
[`mps_wire_probability_dispatch_a800.json`](../../benchmarks/results/local/mps_wire_probability_dispatch_a800.json)
artifact preserves 30 synchronized groups of 100 invocations for each case on
`jp-a800-171` and `jp-a800-172`, under stock Triton 3.7.1 and FlagTree 0.7.0.
Across the fixed sequential-sampling shape matrix, the direct kernel wrapper is
`2.14x` to `3.42x` faster than the equivalent PyTorch reduction, with maximum
absolute error `8.94e-8`. After removing redundant normalization and replacing
per-wire host synchronization with device-side asynchronous validation, the
complete public dispatch path is `1.27x` to `1.40x` faster across all 16
host/compiler/shape cases. The canonical aggregate records
`eligible_for_default`, and MPS-007 is now a `provisional` implementation with
default dispatch inside the measured window and the explicit kill switch
above. This is bounded development-hardware evidence, not a release gate or
scalability claim.
Reproduce or validate it with
[`benchmarks/mps_wire_probability_dispatch.py`](../../benchmarks/mps_wire_probability_dispatch.py).

NUM-002 contracts the explicit non-view layout `azcb,czdb->zad` as a strided
complex batched matrix multiplication, avoiding canonical input
materialization. The checked-in
[`tn_layout_contraction_a800.json`](../../benchmarks/results/local/tn_layout_contraction_a800.json)
artifact preserves 30 synchronized groups of 10 invocations for each of four
fixed contraction shapes on `jp-a800-171` and `jp-a800-172`, under stock
Triton 3.7.1 and FlagTree 0.7.0. Runtime dispatch selects the catalog kernel
only for forward-only complex64 CUDA calls with that exact equation, explicit
input shapes `(64, 16, 64, 16)` by `(64, 16, 64, 16)`, and logical BMM shape
`(16, 64, 1024, 64)`. Requests outside that measured signature, including all
gradient-bearing calls, return directly to native `torch.einsum` before layout
analysis.

Across the four selected host/compiler cases, public dispatch is `1.066x` to
`1.790x` faster than native einsum. Across the 12 fallback cases it is `0.946x`
to `1.225x`, with at most `3.24 us` positive wrapper overhead. The evidence
contract accepts a fallback only when it retains at least `0.95x` relative
performance or adds no more than `5 us` absolute overhead. Direct forward plus
backward ranges from `0.289x` to `1.057x` and does not win across the matrix, so
training remains on the native path. Maximum forward absolute and relative L2
error are `2.22e-4` and `8.31e-7`; maximum gradient absolute and relative L2
error are `6.10e-5` and `4.27e-7`.

The canonical aggregate records `retain_current_policy` for this exact narrow
window. NUM-002 remains `experimental`; the result does not authorize shape
extrapolation, maturity promotion, a release gate, or a scalability claim.
Reproduce or validate it with
[`benchmarks/tn_layout_contraction.py`](../../benchmarks/tn_layout_contraction.py).

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

The error-mitigation row is the one family carrying a workload that already runs
without a fused kernel of its own: zero-noise extrapolation is a sequence of
exact expectation reads at scaled noise strengths, so the foundation measurement
and expectation kernels serve it and the unit lives in `flagquantum.algorithms`
rather than here. Readout-error mitigation, probabilistic error cancellation and
Clifford data regression remain planning items with no implementation.

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

The current 26 semantics and 33 implementations are implemented. SV-001-A,
MPS-003 through MPS-007, and MEAS-001 through MEAS-003 are provisional after
their evidenced default-dispatch promotions; the other 24 implementations
remain experimental.
The rest of the 100/800 portfolio is planned or candidate work, not shipped
capability.

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
4. expand direct FlagTree implementations only where TLE or FlagTree-owned
   source provides cross-project value;
5. expand foundation semantics from measured workload traces;
6. promote recurring application structures only after end-to-end evidence.

This keeps the direct FlagQuantum kernel line primary and independently usable,
while FlagTree and Torch-FL remain compatible provider and integration paths
rather than mandatory ownership layers.
