# Statevector numerics

This package owns dense statevector numerical algorithms. It does not choose
devices, plan shards, perform distributed communication, or assemble Runtime
evidence.

- Start in `local.py` for initial-state construction, the local execution loop,
  dense observables, and sampling.
- Start in `operations.py` for gate application, layout, matrix composition,
  fusion, and rank-local tensor operations.
- The stable user entry points remain `fq.Circuit`, `fq.plan`, and `fq.run`.
