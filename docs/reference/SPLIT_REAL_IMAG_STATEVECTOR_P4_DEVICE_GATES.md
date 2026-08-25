# Device-generated Double-Single split statevector P4

P4 is an explicit accuracy experiment for PyTorch devices with reliable FP32
operators but no usable FP64 or complex128 kernels. It preserves P3's four-word
complex state representation and additionally generates supported gates using
device-resident FP32 Double-Single arithmetic.

P4 is not selected by the default runtime and does not change existing CPU,
CUDA, or FlagOS execution paths.

## Device gate-generation boundary

The supported fixed gates are I, X, Y, Z, H, S, Sdg, T, Tdg, SX, SXdg, CX,
CY, CZ, and SWAP. RX, RY, RZ, RXX, RYY, and RZZ accept a direct scalar angle
whose absolute value is at most 1024. Range reduction to the nearest multiple
of pi/2 and sine/cosine evaluation use Double-Single FP32 arithmetic and Horner
polynomials on the logical device.

Device-resident FP32 tensors and `DoubleSingleTensor` values stay on device.
Python numeric values are necessarily ingested by the host and begin with only
FP32 parameter precision; this is reported as
`host_scalar_parameter_ingestion=true`, not hidden as device-to-host fallback.
Float64 tensors, parameter expressions, custom matrices, and out-of-range
angles fail closed.

Every result records these boundaries:

- `device_only_double_single_trigonometry=true`;
- `host_gate_encoding=false`;
- `parameter_host_fallback=false`;
- `state_host_fallback=false`;
- `complex_accelerator_tensor_materialized=false`;
- `host_sync_safety_checks=true`.

## Usage

```python
import torch
import flagquantum as fq

theta = fq.Parameter("theta")
circuit = fq.Circuit(2).h(0).ry(1, theta=theta).cx(0, 1)
observable = fq.algorithms.pauli_term(1.0, "ZZ", (0, 1))

result = (
    fq.experimental.parameter_shift_split_real_imag_device_double_single_gradient(
        circuit,
        observable,
        parameter_bindings={
            "theta": torch.tensor(0.23, dtype=torch.float32, device="cpu")
        },
        device="cpu",
    )
)
print(result.cpu_float64())
```

Diagnostic reconstruction to float64/complex128 occurs only after result words
are transferred to CPU.

## Contract and conformance

The executable numerical plan is returned by
`fq.experimental.split_real_imag_p4_precision_plan()`. The machine-readable
scope is
[`split-real-imag-statevector-p4-device-double-single-contract.toml`](../../split-real-imag-statevector-p4-device-double-single-contract.toml),
and the required operator surface is
[`split_real_imag_statevector_p4_device_double_single.json`](../../flagquantum/runtime/profiles/split_real_imag_statevector_p4_device_double_single.json).

Run CPU conformance with:

```bash
python -c 'import flagquantum as fq; r = fq.experimental.run_split_real_imag_device_double_single_conformance("cpu"); r.require_accepted(); print(r.to_dict())'
```

Run Torch-FL logical-device conformance with:

```bash
python tools/validate_split_real_imag_device_double_single_flagos.py \
  --device flagos:0
```

The bounded matrix covers full state, norm, expectation, and gradient checks at
depths 8, 32, and 128 for seeds 0 and 7, plus a depth-512 state-stability case.
It also measures trigonometric and gate-entry error against CPU float64 and
complex128 diagnostics.

The checked-in A800 native-CUDA and `flagos:0` portability record is
[`split_real_imag_device_double_single_a800_20260825.json`](../../artifacts/split_real_imag_device_double_single_a800_20260825.json).
It is fail-closed by
`tools/validate_split_real_imag_device_double_single_evidence.py`. The FlagOS
route is a single-device Torch-FL test; it neither validates FlagCX collectives
nor makes a distributed-execution claim.

## Deliberate limitations

P4 remains a correctness-first eager implementation. It has no fused kernels,
native autograd, optimizer API, compilation, distributed execution, automatic
precision selection, algorithmic convergence certification, provider-internal
route audit, domestic-hardware certification, performance claim, or production
claim. CUDA and CUDA-backed `flagos:0` results are portability evidence only.
Double-Single retains the FP32 exponent range and is not generally equivalent
to IEEE binary64 or complex128.
