# Double-Single FP32 numerical foundation

FlagQuantum provides an experimental, device-generic Double-Single foundation
under `flagquantum.numerics`. A real value is represented as the unevaluated
sum of two `torch.float32` tensors, conventionally called the high and low
words. Error-free transformations retain arithmetic residuals that ordinary
FP32 discards.

This is software-extended precision, not neural-network quantization and not a
claim that FP32 is generally equivalent to FP64 or complex128. Double-Single
retains the FP32 exponent range and offers roughly 48 significant bits only
within the bounded arithmetic implemented and certified here.

## Current scope

The foundation implements:

- Knuth `two_sum` and Dekker `quick_two_sum`;
- binary32 splitting with splitter 4097 and residual-preserving products;
- normalized real add, subtract, multiply, sum, and dot;
- Newton-refined reciprocal, reciprocal square root, and square root;
- bounded device-side sine and cosine with Double-Single range reduction;
- split-real/imaginary complex add, multiply, and magnitude squared;
- PyTorch-composed autograd and device preservation;
- deterministic CPU complex128/float64 conformance probes.

Run the machine-readable certification through the explicitly unstable API:

```python
import flagquantum as fq

report = fq.experimental.run_double_single_conformance("cpu")
report.require_accepted()
print(report.to_json())
```

The authoritative algorithm, threshold, and unsupported-scope declaration is
`contracts/double-single-contract.toml`. The CPU conformance suite covers catastrophic
cancellation in sums and dot products plus a 2048-step complex phase chain.
Each candidate must meet its absolute error bound. It must also improve on the
device's native FP32 baseline by the declared factor unless that baseline is
already within the same bound, in which case no regression is required.
Scheduled accelerator CI runs the same FP32 operations on CUDA when a hardware
runner is available; that is portability evidence, not domestic-card
certification.

## Selective P2 runtime integration

The explicit split real/imag P2 precision executor now composes these
primitives for Pauli inner products, Hamiltonian term sums, and
parameter-shift gradient accumulation. State storage, gate generation, and
gate application remain ordinary split FP32. P2 returns both high and low
words and reconstructs float64 only after moving diagnostic values to CPU.
See
[`SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md`](SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md).

P3 extends the same primitives to four-word complex state storage, eager gate
application, periodic normalization, observables, and parameter-shift gradients.
It explicitly uses CPU complex128 gate encoding before transferring FP32 words
to the execution device. See
[`SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md`](SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md).

P4 removes P3's host gate-encoding boundary for the built-in fixed gates and
RX/RY/RZ/RXX/RYY/RZZ with direct scalar angles satisfying `|angle| <= 1024`.
Range reduction, sine/cosine, gate construction, and execution use FP32
Double-Single words on the logical execution device. See
[`SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md`](SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md).

None of P2, P3, or P4 is an automatic precision provider.

## Deliberate limitations

The default statevector runtime continues to reject a Double-Single precision
plan. P4 is explicit, bounded, and experimental; it does not provide unbounded
transcendentals, parameter expressions, optimized/fused dispatch, distributed
collectives, decomposition kernels, optimizer master state, compiled
execution, Torch-FL route auditing, or vendor certification.

Inputs to Dekker splitting must be finite and small enough that multiplication
by 4097 does not overflow. Runtime integration must add range monitoring,
representation-specific gate kernels, end-to-end state/gradient/convergence
evidence, and an explicit precision provider before the existing fail-closed
guard can be relaxed.
