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
`double-single-contract.toml`. The CPU conformance suite covers catastrophic
cancellation in sums and dot products plus a 2048-step complex phase chain.
Each candidate must meet its absolute error bound. It must also improve on the
device's native FP32 baseline by the declared factor unless that baseline is
already within the same bound, in which case no regression is required.
Scheduled accelerator CI runs the same FP32 operations on CUDA when a hardware
runner is available; that is portability evidence, not domestic-card
certification.

## Deliberate limitations

The default statevector runtime continues to reject a Double-Single precision
plan. This PR does not provide gate dispatch, transcendental gate generation,
distributed collectives, decomposition kernels, optimizer master state,
checkpoint encoding, Torch-FL integration, or vendor certification. It also
does not certify compiled execution and does not silently fall back to CPU or
float64.

Inputs to Dekker splitting must be finite and small enough that multiplication
by 4097 does not overflow. Runtime integration must add range monitoring,
representation-specific gate kernels, end-to-end state/gradient/convergence
evidence, and an explicit precision provider before the existing fail-closed
guard can be relaxed.
