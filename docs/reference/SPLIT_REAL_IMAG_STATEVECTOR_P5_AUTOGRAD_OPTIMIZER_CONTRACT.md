# P5 autograd and optimizer contract

P5 now provides an experimental **CPU-only PyTorch autograd bridge** over the
P4 device-generated Double-Single executor. The precision-preserving optimizer
lane remains a contract and is not implemented. The default FlagQuantum
runtime is unchanged.

## Available CPU autograd bridge

The bridge accepts direct named scalar `torch.float32` parameters, returns a
scalar FP32 loss, and supplies first-order gradients through a custom
`torch.autograd.Function`:

```python
import torch

import flagquantum as fq
from flagquantum.algorithms import pauli_term

theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
loss = fq.experimental.split_real_imag_device_double_single_autograd_expectation(
    fq.Circuit(1).ry(0, theta=fq.Parameter("theta")),
    pauli_term(1.0, "Z", (0,)),
    parameter_bindings={"theta": theta},
    device="cpu",
)
loss.backward()
```

The first slice supports direct named parameters on RX, RY, RZ, RXX, RYY, and
RZZ; bounded real Pauli Hamiltonians; batch size one; and a scalar expectation
output. Higher-order and compiled autograd are rejected rather than silently
falling back.

## Precision boundary

P4 represents an expectation and parameter-shift gradient with two FP32 words.
A normal FP32 PyTorch leaf stores one FP32 word in `.grad`; that tensor cannot
honestly be relabeled as a Double-Single gradient. The bridge therefore:

1. evaluates the forward expectation with P4 Double-Single arithmetic;
2. evaluates the backward pass with P4 explicit high/low parameter shift; and
3. rounds once to FP32 when returning the scalar loss and PyTorch `.grad`.

The reported PyTorch boundary is `float32_boundary`. End-to-end Double-Single
gradient precision is not claimed. The future precision optimizer lane will
instead consume the explicit high/low gradient and maintain a high/low master
parameter, initially using Double-Single SGD without momentum or weight decay.

## CPU numerical evidence

The checked conformance workload compares depths 8, 32, and 128 with seeds 0
and 7 against independent CPU complex128 parameter-shift and finite-difference
diagnostics. The current six-case result passes with:

- maximum forward absolute error: `1.343e-8`;
- maximum gradient relative error: `3.508e-8`;
- minimum gradient cosine similarity: `0.9999999999999996`; and
- maximum finite-difference gradient relative error: `1.861e-7`.

These are CPU reference results, not accelerator, optimizer-trajectory,
convergence, performance, or production evidence.

## Remaining scope

The precision optimizer, parameter expressions, trainable Hamiltonian
coefficients, standard optimizer equivalence, Adam/AdamW, momentum, weight
decay, mixed-precision loss scaling, CUDA, Torch-FL `flagos:0`, distributed
execution, and automatic runtime selection remain unavailable in P5.
FlagCX collectives are explicitly excluded; a distributed claim requires a
later contract and real distributed evidence.

The authoritative machine-readable declaration is
[`split-real-imag-statevector-p5-autograd-optimizer-contract.toml`](../../split-real-imag-statevector-p5-autograd-optimizer-contract.toml).
