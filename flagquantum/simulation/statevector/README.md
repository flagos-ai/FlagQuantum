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
- Start in `double_single.py` for full Double-Single state evolution,
  normalization, and observable reduction.
- Start in `double_single_host_gates.py` for explicit CPU reference gate
  encoding.
- Start in `double_single_device_gates.py` for device-resident FP32 gate
  encoding without host fallback.
- Start in `small.py` for specialized exact kernels limited to very small,
  deep statevector workloads.
- The stable user entry points remain `fq.Circuit`, `fq.plan`, and `fq.run`.
