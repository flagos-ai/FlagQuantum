# P5 autograd and optimizer contract

P5 is currently a design contract, not an implemented runtime capability. It
defines how FlagQuantum may connect P4's Double-Single execution to PyTorch
autograd and precision-preserving optimizer updates without overstating the
precision of a standard `Tensor.grad`.

## The precision boundary

P4 represents a scalar gradient as two FP32 words. A normal FP32 PyTorch leaf
stores one FP32 word in `.grad`; that object cannot be relabeled as a
Double-Single gradient. P5 therefore separates two lanes:

1. A PyTorch autograd bridge may expose a conventional FP32 `.grad` for normal
   training-loop interoperability. Its reported precision remains
   `float32_boundary` until a mathematically valid paired-gradient
   representation exists and passes gradcheck.
2. The precision optimizer consumes the explicit high/low parameter-shift
   result and updates a high/low parameter master state. Its initial scope is
   Double-Single SGD without momentum, weight decay, or mixed-precision loss
   scaling.

This separation prevents a convenient PyTorch API from silently discarding the
low word while claiming end-to-end Double-Single gradients.

## Initial implementation scope

The future implementation is limited to direct named device-FP32 parameters on
RX, RY, RZ, RXX, RYY, and RZZ; bounded real Pauli Hamiltonians; batch size one;
and a scalar expectation output. The default runtime remains unchanged.

Parameter expressions, trainable Hamiltonian coefficients, standard optimizer
equivalence, Adam/AdamW, momentum, weight decay, compiled autograd,
distributed execution, and automatic runtime selection are outside the first
implementation slice.

## Evidence required before capability claims

Implementation work must provide:

- CPU complex128 and finite-difference diagnostics;
- PyTorch gradcheck for the declared autograd boundary;
- depths 8, 32, and 128 with seeds 0 and 7;
- one-, 16-, and 64-step optimizer trajectory comparisons;
- two-qubit VQE and three-qubit QAOA trajectory workloads;
- native CUDA and single-device Torch-FL `flagos:0` evidence.

Trajectory agreement is numerical integration evidence, not algorithmic
convergence certification. FlagCX collectives are explicitly excluded from P5;
a distributed claim requires a later contract and real distributed evidence.

The authoritative machine-readable declaration is
[`split-real-imag-statevector-p5-autograd-optimizer-contract.toml`](../../split-real-imag-statevector-p5-autograd-optimizer-contract.toml).
