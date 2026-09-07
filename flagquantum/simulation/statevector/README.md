# Statevector numerics

This package owns dense statevector numerical algorithms. It does not choose
devices, plan shards, perform distributed communication, or assemble Runtime
evidence.

- Start in `local.py` for initial-state construction, the local execution loop,
  dense observables, and sampling.
- Start in `operations.py` for gate application, layout, matrix composition,
  fusion, and rank-local tensor operations.
- Start in `adjoint.py` for local adjoint and gradient primitives.
- Start in `noisy.py` for one already-lowered batch of noisy trajectories.
- Start in `split_real_imag.py` for split-storage FP32 and selective
  Double-Single numerical primitives.
- The stable user entry points remain `fq.Circuit`, `fq.plan`, and `fq.run`.
