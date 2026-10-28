# Statevector numerics

This package owns dense statevector numerical algorithms. It does not choose
devices, plan shards, perform distributed communication, or assemble Runtime
evidence.

- Start in `local.py` for initial-state construction, the local execution loop,
  dense observables, and sampling.
- Start in `operations.py` for gate application, layout, matrix composition,
  fusion, and rank-local tensor operations.
- Start in `batching.py` for parameter windows and rotation-region angle packing.
  Compiled real scalar slots with matching dtype/device are packed together;
  other forms retain the existing per-gate parameter conversion and validation.
- Start in `two_qubit_cpu.py` for the CPU kernels that apply a dense two-qubit
  gate directly to the contiguous four-amplitude blocks of an index-adjacent
  qubit pair.
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

For a small parameter-binding change, run
`python -m pytest tests/unit/test_statevector_bound_rotation_regions.py -q`.
It exercises ordinary Module Adam updates with shared nonlinear parameters,
trainable features and scalar broadcasting against the unpacked implementation,
plus vector/cast/constant handling and malformed batch refusal. Use the existing
`python -m benchmarks.module_execution_matrix --output /tmp/fq-module-matrix`
for public-entry timing; an internal packing microbenchmark alone does not
establish ordinary training performance.
