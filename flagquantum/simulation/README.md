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

Keep ordinary numerical changes inside this directory. A change that also
requires Compiler or Runtime policy should be split at the existing contract
boundary before implementation.
