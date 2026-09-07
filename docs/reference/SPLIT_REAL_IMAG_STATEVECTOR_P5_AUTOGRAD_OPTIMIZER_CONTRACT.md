# P5 autograd and Double-Single SGD

> Internal development reference. These lanes are evidence implementations,
> not public FlagQuantum SDK APIs.

P5 provides two experimental training lanes over the P4 device-generated
Double-Single executor. The autograd bridge remains CPU-only. The explicit SGD
lane has single-device CPU, native CUDA, and CUDA-backed Torch-FL `flagos:0`
portability evidence. The default FlagQuantum runtime is unchanged.

## PyTorch autograd interoperability lane

The autograd bridge accepts direct named scalar `torch.float32` parameters,
returns a scalar FP32 loss, and supplies first-order gradients through a custom
`torch.autograd.Function`:

```python
import torch

import flagquantum as fq
from flagquantum.algorithms import pauli_term
from flagquantum.runtime.backends.statevector.split_real_imag_autograd import (
    split_real_imag_device_double_single_autograd_expectation,
)

theta = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
loss = split_real_imag_device_double_single_autograd_expectation(
    fq.Circuit(1).ry(0, theta=fq.Parameter("theta")),
    pauli_term(1.0, "Z", (0,)),
    parameter_bindings={"theta": theta},
    device="cpu",
)
loss.backward()
```

P4 computes the forward expectation and explicit parameter-shift backward with
high/low words, but a normal FP32 PyTorch leaf can store only one word in
`.grad`. The returned loss and `.grad` are therefore reported as an explicit
`float32_boundary`, never as end-to-end Double-Single gradients.

## Precision-preserving optimizer lane

The Double-Single SGD lane bypasses `.grad`. It consumes P4's explicit
high/low parameter-shift result and maintains a functional high/low master
parameter state:

```python
import torch

from flagquantum.runtime.backends.statevector.split_real_imag_autograd_optimizer import (
    initialize_split_real_imag_double_single_sgd,
    split_real_imag_double_single_sgd_step,
)
from flagquantum.simulation.numerics.double_single import DoubleSingleTensor

device = torch.device("cpu")
master = torch.tensor(0.23, dtype=torch.float64)
state = initialize_split_real_imag_double_single_sgd(
    {"theta": DoubleSingleTensor.from_float64(master).to(device)}
)
learning_rate = DoubleSingleTensor.from_float64(
    torch.tensor(0.1, dtype=torch.float64)
).to(device)
result = split_real_imag_double_single_sgd_step(
    fq.Circuit(1).ry(0, theta=fq.Parameter("theta")),
    pauli_term(1.0, "Z", (0,)),
    state,
    learning_rate=learning_rate,
)
state = result.state
```

Each update evaluates `parameter - learning_rate * gradient` with
Double-Single multiplication, subtraction, and renormalization. The first
algorithm is plain SGD without momentum, weight decay, loss scaling, or
`torch.optim.Optimizer`/`state_dict` compatibility.

The parameter words, gradient words, and learning-rate tensor must already
reside on the same execution device. Initialization may encode and transfer
them once, but a training step rejects Python learning-rate scalars and
cross-device tensors so the steady-state loop cannot introduce per-step host
tensor transfers.

## CPU numerical evidence

Autograd conformance covers depths 8, 32, and 128 with seeds 0 and 7 against
CPU complex128 parameter-shift and finite-difference diagnostics. Its maximum
gradient relative error is `3.508e-8`, which correctly reflects the final FP32
delivery boundary rather than a Double-Single `.grad` claim.

Optimizer conformance compares Double-Single and FP32 master-parameter SGD
against complex128 trajectories at steps 1, 16, and 64:

| Workload | Step | DS parameter error | FP32 parameter error | DS loss error | FP32 loss error |
| --- | ---: | ---: | ---: | ---: | ---: |
| two-qubit VQE | 1 | `7.57e-11` | `6.57e-9` | `1.44e-11` | `1.03e-9` |
| two-qubit VQE | 16 | `1.49e-9` | `3.84e-8` | `2.61e-10` | `5.92e-9` |
| two-qubit VQE | 64 | `1.24e-8` | `1.12e-7` | `3.32e-9` | `3.01e-8` |
| three-qubit QAOA | 1 | `6.54e-10` | `1.24e-8` | `8.84e-10` | `9.59e-9` |
| three-qubit QAOA | 16 | `9.40e-9` | `1.71e-8` | `8.16e-9` | `1.42e-8` |
| three-qubit QAOA | 64 | `3.72e-10` | `7.84e-8` | `2.22e-12` | `4.67e-10` |

A cancellation case also applies 64 updates of `2^-31` to a parameter at
`1.0`: the FP32 master loses all updates (`2.98e-8` error), while the
Double-Single master matches the float64 reference exactly in the diagnostic.
Every checked trajectory point requires Double-Single to be no worse than the
FP32 master as well as satisfying the absolute CPU envelope.

These results demonstrate bounded numerical trajectory agreement and recovery
of sub-ULP parameter updates. They are not algorithmic convergence,
performance, or production certification.

## A800 CUDA and Torch-FL reference

The exact source revision `8671e660` passed the same 1/16/64-step optimizer
matrix on an NVIDIA A800-SXM4-80GB through both `cuda:0` and Torch-FL
`flagos:0`. High and low master words stayed on the selected logical device,
`Tensor.grad` was not used, and float64 tensors were not materialized on the
accelerator. The two routes produced zero recorded metric delta.

This is CUDA and Torch-FL portability evidence only. It does not audit
Torch-FL's provider-internal route, validate FlagCX, certify a domestic
accelerator, establish FP64/complex128 equivalence, or prove convergence.
The machine-readable evidence is
[`split_real_imag_optimizer_a800_20260825.json`](../../artifacts/split_real_imag_optimizer_a800_20260825.json).

## Scope and remaining gates

Both lanes support direct named parameters on RX, RY, RZ, RXX, RYY, and RZZ;
bounded real Pauli Hamiltonians; batch size one; and scalar expectations.
Parameter expressions, trainable coefficients, Adam/AdamW, momentum, weight
decay, mixed-precision loss scaling, higher-order or compiled autograd,
distributed execution, and automatic selection remain unavailable. FlagCX is
explicitly excluded until a later distributed contract has real collective
evidence. CUDA and `flagos:0` are explicit optimizer routes, not automatic
runtime selections or hardware certifications.

The authoritative machine-readable declaration is
[`split-real-imag-statevector-p5-autograd-optimizer-contract.toml`](../../contracts/split-real-imag-statevector-p5-autograd-optimizer-contract.toml).
