# Split real/imag FP32 local statevector P0

FlagQuantum provides an experimental single-device statevector executor that
stores amplitudes as two independent `torch.float32` tensors. The real and
imaginary parts, gate generation, and matrix arithmetic remain FP32 on the
selected PyTorch device. This allows a forward path to be evaluated on devices
whose complex dtype coverage is missing or incomplete.

This is an explicit experimental API. The ordinary `statevector` runtime and
its CUDA behavior are unchanged, and no precision plan selects this executor
automatically.

```python
import flagquantum as fq

circuit = fq.Circuit(3).h(0).ry(1, theta=0.27).cx(0, 2)
result = fq.experimental.execute_split_real_imag_statevector(
    circuit.to_ir(), device="cpu"
)

assert result.real.dtype.is_floating_point
assert result.imag.dtype.is_floating_point
assert result.summary()["representation"] == "split_real_imag"
```

`result.cpu_complex128()` exists only for diagnostics and reference comparison.
It first transfers both FP32 words to CPU and then constructs a complex128
tensor. The execution path itself never needs a complex accelerator tensor.

## P0 contract

The authoritative machine-readable boundary is
[`split-real-imag-statevector-contract.toml`](../../contracts/split-real-imag-statevector-contract.toml).
P0 supports a fixed set of built-in one- and two-qubit gates, scalar bound
parameters, a batch-one `|0...0>` local statevector, and CPU-reference conformance at
depths 8, 32, and 128. Its operator requirements are packaged in
[`split_real_imag_statevector_p0.json`](../../flagquantum/runtime/profiles/split_real_imag_statevector_p0.json).

Before state allocation, the executor resolves the requested platform and runs
the float32 profile probes. Unsupported operators fail closed. Result metadata
records the profile hash, evidence IDs, logical device, provider, storage and
compute dtype, and the `single_device_fast_path` classification.

Run CPU conformance with:

```bash
python -c 'import flagquantum as fq; r = fq.experimental.run_split_real_imag_conformance("cpu"); r.require_accepted(); print(r.to_dict())'
```

In a Torch-FL CUDA-reference environment, run:

```bash
python tools/validate_split_real_imag_flagos.py --device flagos:0
```

That validator proves execution through the logical `flagos:0` device and
checks numerical behavior against CPU complex128. It does not prove that a
provider internally avoided every host-mediated implementation. Such a claim
requires Torch-FL route/profiler evidence, so the report deliberately sets
`provider_internal_route_audited=false` and `hardware_certification=false`.

## Relationship to Double-Single

This executor uses ordinary FP32 for both words of a complex amplitude; it is
not Double-Single and does not emulate complex128. The next numerical step is
to measure sensitive reductions and selected gate kernels, then introduce
Double-Single only where those measurements require it. Keeping representation
portability separate from precision escalation makes both paths reviewable.

P1 adds an independent, opt-in Pauli expectation and parameter-shift layer on
top of this representation. See
[`SPLIT_REAL_IMAG_STATEVECTOR_P1.md`](SPLIT_REAL_IMAG_STATEVECTOR_P1.md). P0
itself remains forward-only and continues to reject trainable parameters.

## Deliberate limitations

P0 rejects custom initial states, batches larger than one, custom matrices, and
trainable parameters. Gradients, optimizer
updates, sampling, the observables API, checkpointing, compilation, distributed
execution, and automatic runtime selection are unsupported. It neither changes
CUDA defaults nor introduces Torch-FL as a core installation dependency.
