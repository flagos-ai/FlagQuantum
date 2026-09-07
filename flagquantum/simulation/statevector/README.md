# Statevector numerics

This package owns dense statevector numerical algorithms. It does not choose
devices, plan shards, perform distributed communication, or assemble Runtime
evidence.

- Start in `local.py` for initial-state construction, the local execution loop,
  dense observables, and sampling.
- Gate application, layout, matrix composition, and fusion remain in
  `../statevector_ops.py` until their own verified migration slice.
- The stable user entry points remain `fq.Circuit`, `fq.plan`, and `fq.run`.
