# Full Double-Single split statevector P3

> Internal development reference. P3 is not an SDK API or an automatic runtime
> provider.

P3 is an explicit accuracy experiment for devices whose PyTorch backend offers
reliable FP32 operators but no usable FP64 or complex128 kernels. Each complex
amplitude is stored as four device-resident FP32 words: real high/low and
imaginary high/low. Gate application, periodic normalization, Pauli reduction,
Hamiltonian accumulation, and parameter-shift accumulation all retain
Double-Single residuals.

P3 is not selected by the default runtime and does not change the existing CPU
or CUDA executors.

## Gate encoding boundary

P3 does not yet implement device-only Double-Single sine and cosine. Parameters
and gate matrices are generated with CPU float64/complex128, split into FP32
high/low words, and transferred to the requested device before state evolution.
The state is never moved to CPU during execution. Every result reports:

- `host_gate_encoding=true`;
- `state_host_fallback=false`;
- `complex_accelerator_tensor_materialized=false`.

Python numeric parameters are encoded directly as CPU float64. An input tensor
already stored as FP32 retains only the precision it actually contains; P3 does
not invent discarded parameter bits.

## Usage

```python
import flagquantum as fq
from flagquantum.runtime.backends.statevector.split_real_imag_double_single import (
    parameter_shift_split_real_imag_double_single_gradient,
)

theta = fq.Parameter("theta")
circuit = fq.Circuit(2).h(0).ry(1, theta=theta).cx(0, 1)
observable = fq.algorithms.pauli_term(1.0, "ZZ", (0, 1))

result = parameter_shift_split_real_imag_double_single_gradient(
    circuit,
    observable,
    parameter_bindings={"theta": 0.23},
    device="cpu",
)
print(result.cpu_float64())
```

The accelerator result remains FP32 high/low words. Diagnostic reconstruction
occurs only after all words are transferred to CPU.

## Precision and normalization

The internal executable plan is returned by
`split_real_imag_p3_precision_plan()`. P3 normalizes the state
every 16 gates by default using a Double-Single norm and two Newton refinements
of an FP32 reciprocal-square-root seed. Set `renormalize_every=0` to disable
periodic normalization or a positive interval to make the policy explicit.

The machine-readable contract is
[`split-real-imag-statevector-p3-double-single-contract.toml`](../../contracts/split-real-imag-statevector-p3-double-single-contract.toml),
and the operator surface is
[`split_real_imag_statevector_p3_double_single.json`](../../flagquantum/runtime/profiles/split_real_imag_statevector_p3_double_single.json).

## Conformance

The bounded matrix covers depths 8, 32, and 128 with seeds 0 and 7. It compares
the full P3 state, norm, Pauli expectation, and parameter-shift gradient with a
CPU complex128 reference and compares end-to-end expectation/gradient error
with P2:

```bash
pytest tests/test_split_real_imag_double_single_conformance.py
```

In a Torch-FL CUDA-reference environment:

```bash
python tools/validate_split_real_imag_double_single_flagos.py --device flagos:0
```

The checked-in A800 native-CUDA and `flagos:0` portability record is
[`split_real_imag_double_single_a800_20260824.json`](../../artifacts/split_real_imag_double_single_a800_20260824.json).
It is fail-closed by
`tools/validate_split_real_imag_double_single_evidence.py` and does not promote
the experimental path to a hardware, convergence, performance, vendor, or
production certification.

This is depth-stability and portability evidence, not algorithmic convergence
certification. CUDA-backed `flagos:0` does not certify a domestic accelerator
or audit Torch-FL's provider-internal route.

## Deliberate limitations

P3 is a correctness-first eager implementation. It uses scalar loops over the
2x2 and 4x4 gate dimensions and has no fused kernel, native autograd, optimizer
API, compilation, distributed execution, device-only transcendental gate
generation, automatic precision selection, performance claim, vendor
certification, or production claim. Double-Single retains FP32 exponent range
and is not generally equivalent to IEEE binary64 or complex128.
