# Runtime Executors

This package executes an already selected plan. It owns execution lifecycle,
device and rank coordination, collectives, checkpoints, gradients, and evidence.
Numerical state updates and tensor operations belong in `flagquantum.simulation`;
external QPU and service adaptation belongs in `flagquantum.compute.remote`.

Start in the representation subpackage named by the plan: `statevector/`,
`mps/`, `tensor_network/`, or `jax/`. A typical executor change should remain in
that subpackage and use existing Core contracts and Simulation kernels.


Node counts and torchrun local-size resolution come directly from
`runtime/distributed/placement.py`, the backend-neutral owner. Keep explicit
JAX device/profile policy in `jax/runtime_environment.py`, subgroup handling
and lifecycle in their existing owners, and each communication-tier calculation
with its current caller. Do not recreate count formulas or internal forwarding
aliases in individual executors.
