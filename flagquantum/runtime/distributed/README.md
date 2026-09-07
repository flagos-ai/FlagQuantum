# Distributed Runtime Foundations

This directory owns distributed concerns shared by every execution backend:
process-group context and lifecycle, backend-selection policy, executor
protocols, communication identity, transport observability, conformance, and
cross-backend capability evidence.

It does not own state partitioning, numerical kernels, gradient algorithms, or
backend-specific execution. Statevector code belongs in
`runtime/backends/statevector`, MPS code in `runtime/backends/mps`, and tensor
network code in `runtime/backends/tensor_network`. JAX-specific orchestration
belongs in `runtime/backends/jax`.

## Ten-minute change path

- Change process-group setup, rank placement, or teardown in `context.py`.
- Change development and production backend selection in `backend_policy.py`.
- Change backend-neutral execution requests and records in `protocols.py`.
- Change communication-route identity in `identity.py` and observable transport
  facts in `transport_observability.py`.
- Change FlagOS activation only in `flagos_runtime.py`.
- Change evidence acceptance in `conformance.py`, the relevant `*_profile.py`,
  or `workload_capability.py`; do not add algorithm execution there.

Start with the focused distributed tests, then run the CPU vertical slice and
architecture checks. If a change mentions amplitudes, MPS bonds, tensor
contractions, or a framework-specific array, it belongs in a backend directory.
