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
private layouts, gate application, fusion, and tensor operations. Neither file
is a new public API. `Circuit.state()` remains the stable user facade and
Runtime enters through `run_local_statevector()`.

`density_matrix.py` owns local exact density evolution, Kraus application, and
density-matrix measurements. Runtime owns noise lowering and execution-plan
dispatch through `runtime/noise_registry.py`.

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
compile cache. `mps_factorization.py` owns QR/SVD numerical routines, and
`mps_reverse.py` owns rank-local adjoint projection and VJP evaluation.
Distributed ownership, transport ordering, memory budgets, microbatch
selection, checkpointing, and evidence remain in Runtime.

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
site behavior, start in `mps_site_kernels.py`; for QR/SVD behavior, start in
`mps_factorization.py`; for local VJP behavior, start in `mps_reverse.py`.
Run:

```bash
python -m pytest tests/unit/test_mps_site_kernels.py \
  tests/unit/test_issue105_dynamic_bond_compile_cache.py \
  tests/unit/test_mps_reverse_numerics.py \
  tests/unit/test_issue052_mps_training.py -q
```

Keep ordinary numerical changes inside this directory. A change that also
requires Compiler or Runtime policy should be split at the existing contract
boundary before implementation.
