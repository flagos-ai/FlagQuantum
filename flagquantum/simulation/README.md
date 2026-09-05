# Simulation

This directory owns FlagQuantum's numerical simulation algorithms: dense
statevector, MPS, tensor-network, noise evolution, and their differentiable
kernels.

It does not own user policy, device selection, distributed lifecycle, provider
identity, fallback decisions, durable jobs, or public result assembly. Those
belong to Runtime and Providers. Simulation consumes Core IR and operator
semantics and may use PyTorch or isolated accelerator kernels.

## Local statevector path

`statevector.py` owns the local execution loop and `statevector_ops.py` owns its
private layouts, gate application, gate-matrix composition, fusion, and tensor
operations, including compressed-to-local basis index expansion. Neither file
is a new public API. `Circuit.state()` remains the
stable user facade and Runtime enters through `run_local_statevector()`.

`statevector_adjoint.py` owns local adjoint numerical primitives that do not
depend on shard ownership or communication: supported rotation derivatives and
the real-valued complex inner product. Runtime retains shard indexing, chunk
policy, rematerialization, collectives, communication, and backward evidence.

`density_matrix.py` owns local exact density evolution, Kraus application, and
density-matrix measurements. Compiler owns noise lowering; Runtime owns
execution-plan dispatch through `runtime/noise_registry.py`.

`noisy_statevector.py` owns batched gate application, Kraus sampling,
amplitude-damping evolution, normalization, and Z-expectation numerics for the
statevector trajectory backend, including the instruction loop for one
already-lowered trajectory batch. Runtime retains trajectory IDs and
random-stream construction, readout-error handling, convergence, retry,
checkpointing, collectives, and result assembly.

`double_single_host_gates.py` owns P3's explicit CPU reference encoding;
`double_single_device_gates.py` separately owns P4's device-resident FP32
gate-matrix numerics so its no-CPU/no-complex128 rule remains source-auditable.
`double_single_statevector.py` owns the shared gate application and state
normalization numerics used by its P3 and P4 executors.
Runtime retains precision authorization, platform selection, execution evidence,
and conformance reporting.

`split_real_imag_statevector.py` owns P0/P1's FP32 real/imag gate matrices,
gate application, and the local zero-state execution loop. Runtime retains
parameter binding, preflight, platform selection, observables, result
construction, and conformance reporting.

`mps_local.py` owns the single-device, noiseless MPS instruction loop.
`mps_noisy.py` owns the numerical loop for one already-lowered noisy trajectory
and accepts an initialized MPS plus an explicit random generator.
`mps_execution.py` preserves the public wrappers and adapts legacy Circuit
inputs. Multi-trajectory ownership, random streams, convergence, retry,
checkpoint/restart, and rank result merging belong to
`runtime/trajectories/mps.py`. The primary `run_native` path lowers noise once
before entering the lowered MPS entry points; only protected direct legacy
calls still perform lowering in the compatibility wrapper.

`tensor_local.py` owns local tensor-network plan construction and the numerical
state entry point. `tensor_observables.py` owns Pauli/Hamiltonian plan assembly,
MPO compression, and batched observable contraction. `tensor_execution.py`
preserves the public wrappers and amplitude entry points; distributed
scheduling, rank lifecycle, and communication remain outside these paths.

`mps_rank_local.py` owns rank-local MPS instruction dispatch, gate application,
and tensor sizing.
`mps_site_kernels.py` owns eager/compiled site kernels and their bounded
compile cache. `mps_compiled_layers.py` owns equal-shape instruction packing,
batched contraction, and factorization. `mps_factorization.py` owns QR/SVD
numerical routines; `mps_canonicalization.py` owns canonical-site factorization,
transfer absorption, residuals, and center norms; and
`mps_reverse.py` owns reverse pair factorization, truncated-subspace projection,
rank-local adjoint projection, and VJP evaluation.
`mps_observables.py` owns local Pauli-environment, Z/ZZ-channel, and
Heisenberg-MPO scans.
Distributed ownership, transport ordering, memory budgets, microbatch
selection, checkpointing, and evidence remain in Runtime.
Forward preparation and reverse replay both use the same compiled-layer
numerics; Runtime does not rebuild instruction buckets into kernel calls.

For the current migration slice, `Circuit` still owns the initial-state and
lifecycle cache containers. Do not duplicate them here or add a second request
or result model.

## Ten-minute change path

For a local statevector behavior change:

1. start in `statevector.py` for execution order or dispatch;
2. change `statevector_ops.py` only for numerical tensor behavior;
3. run the statevector characterization and CPU vertical-slice tests.

For a local density-matrix change, start in `density_matrix.py` and run:

```bash
python -m pytest tests/test_noise.py -k density_matrix -q
```

For batched noisy-statevector numerics, start in `noisy_statevector.py` and run:

```bash
python -m pytest tests/unit/test_noisy_statevector_numerics.py tests/test_noise.py -q
```

For the local noiseless MPS loop, start in `mps_local.py`; for one lowered noisy
trajectory, start in `mps_noisy.py`. Run:

```bash
python -m pytest tests/test_mps.py -q
```

For local tensor-network plan construction or execution, start in
`tensor_local.py`; for observable behavior, start in `tensor_observables.py`.
Run:

```bash
python -m pytest tests/test_tensor_network.py -q
```

For rank-local distributed-MPS math, start in `mps_rank_local.py`; for compiled
layer numerics, start in `mps_compiled_layers.py`; for compiled site kernels,
start in `mps_site_kernels.py`; for QR/SVD behavior, start in
`mps_factorization.py`; for canonicalization math, start in
`mps_canonicalization.py`; for local VJP behavior, start in `mps_reverse.py`.
For observable contraction math, start in `mps_observables.py`. Run:

```bash
python -m pytest tests/unit/test_mps_site_kernels.py \
  tests/unit/test_mps_compiled_layer_numerics.py \
  tests/unit/test_issue105_dynamic_bond_compile_cache.py \
  tests/unit/test_mps_reverse_numerics.py \
  tests/unit/test_mps_observable_numerics.py \
  tests/unit/test_issue052_mps_training.py -q
```

Keep ordinary numerical changes inside this directory. A change that also
requires Compiler or Runtime policy should be split at the existing contract
boundary before implementation.
