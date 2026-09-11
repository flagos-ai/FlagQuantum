# Runtime Executors

This package executes an already selected plan. It owns execution lifecycle,
device and rank coordination, collectives, checkpoints, gradients, and evidence.
Numerical state updates and tensor operations belong in `flagquantum.simulation`;
external QPU and service adaptation belongs in `flagquantum.remote`.

Start in the representation subpackage named by the plan: `statevector/`,
`mps/`, `tensor_network/`, or `jax/`. A typical executor change should remain in
that subpackage and use existing Core contracts and Simulation kernels.
