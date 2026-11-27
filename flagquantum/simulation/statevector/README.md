# Statevector numerics

This package owns dense statevector numerical algorithms. It does not choose
devices, plan shards, perform distributed communication, or assemble Runtime
evidence.

- Start in `local.py` for initial-state construction, the local execution loop,
  dense observables, and sampling.
- Start in `instruction.py` for gate application, layout, matrix composition,
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

Dynamic trajectory collapse is implemented in `measurement_ops.py`; one static
Instruction acts through `instruction.py`. Runtime supplies its generator and
coordinates feedback/shot lifecycles. Simulation does not choose seeds or
controllers. Forward/reverse local block composition shares the numerical tensor
product in `simulation/gate_matrix.py`; admissibility and scratch evidence stay
with their Runtime sweeps. Start with `tests/test_dynamic_circuit.py` and
`tests/test_dynamic_conformance.py` for the trajectory path.

## Internal helper ownership

Import readout from `measurement_ops.py`, dense/diagonal actions from their
CPU or dispatch modules, and compiled step records from `program.py`.
`local.py` and `operations.py` retain execution/compilation and their actual
dependencies; they do not re-export unused helpers for consumers.
Run the sample-request, single-wire, diagonal-catalog and operations-numerics
scenarios when updating these imports. Existing algorithm and dispatch
assertions remain authoritative.

## Final trajectory sampling

`measurement_ops.py` also owns the single and batched final-trajectory draws.
Runtime supplies the state and explicit generator, then applies readout noise,
frame corrections and feedback/evidence policy. The distinct original
multinomial defaults and output shapes are preserved.

## Weighted observable scalar

`adjoint.py::_weighted_observable_expectation` owns the existing native attempt
and Tensor fallback for an already-owned real diagonal. Cache admission,
allocation, eviction and lifecycle stay in Runtime; the public native provider
and its ABI are unchanged.

## JAX in-process shard updates

`jax/kernels.py::jax_apply_gate_to_shard_arrays` applies an already-admitted
diagonal or dense gate to a complete in-process collection of shard arrays.
It owns basis grouping, diagonal factors, matrix multiplication, functional
updates and the existing missing-amplitude refusal. It neither creates Runtime
shard records nor chooses devices, ranks, transport or claimability. Runtime
validates the gate shape and reconstructs records with the original index and
shard metadata references. This development calculation is not a collective.

Start with `tests/unit/test_jax_shard_gate_numerical_ownership.py` for dense
forward/gradient parity, missing coverage, metadata preservation and replacement
of the numerical implementation without modifying the Runtime consumer.


## Fixed CPU Toffoli

`reversible_cpu.py` owns the exact control-selected target permutation for a
fixed Toffoli. `operations.py` dispatches fixed CPU gates and `local.py` tracks
whether the state buffer belongs to the executor. Only owned, nondifferentiable
states reuse storage; caller inputs and gradient graphs are preserved. An
explicit matrix retains its matrix path and GPU execution retains catalog
admission. Run `python -m pytest tests/unit/test_statevector_reversible_cpu.py -q`
for all four-wire placements, basis states, strided/conjugate gradients and the
public circuit chain. `fq.Circuit(3).x(0).x(1).ccx(0,1,2).state()` has its only
nonzero amplitude at index seven. The original Maxwell workflow is exercised
without changing any phase row, wire, operation or numerical assertion.


## Dense and diagonal matrix selection

Apply one already-built gate matrix with the kernel that fits it.

``_apply_matrix`` lays the state out, permutes the gate's qubits to the front
and permutes back, which a single-qubit gate does not need.
``_apply_diagonal_matrix`` keeps the layout but replaces the batched matmul
with an elementwise multiply, which only a diagonal matrix may take.

``uses_diagonal_kernel`` says the matrix is known to be diagonal, so the
diagonal kernel is available here. It is not a preference: where the
elementwise single-qubit kernel applies as well, that kernel wins, because it
drops the layout permutation the diagonal kernel still pays for, and it
agrees with both of them to under an ulp on every shape measured. It is not
bitwise equal to either, which is why the switch licensing it is off by
default.
