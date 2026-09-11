# Runtime Executor Boundary

This directory owns internal, plan-aware execution lifecycles for statevector,
MPS, tensor-network, and JAX paths. It may coordinate devices, ranks,
collectives, checkpoints, gradients, and evidence, but numerical algorithms
belong in `flagquantum/simulation/`. It is not a public backend API and must not
own credentials, vendor SDK adaptation, or durable job management.
