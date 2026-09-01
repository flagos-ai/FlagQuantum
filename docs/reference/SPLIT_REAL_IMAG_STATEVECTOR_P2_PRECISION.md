# Selective Double-Single split statevector precision P2

P2 is an explicit precision experiment for cancellation-sensitive observable
and gradient reductions on FP32-only PyTorch devices. It deliberately keeps
state amplitudes, gate generation, and gate application as separate real and
imaginary `torch.float32` tensors. Only these operations retain Double-Single
high/low words:

- Pauli inner products;
- Hamiltonian term accumulation;
- parameter-shift gradient accumulation.

```python
import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term

theta = fq.Parameter("theta")
circuit = fq.Circuit(1).ry(0, theta=theta)
observable = Hamiltonian(
    tuple(
        pauli_term(coefficient, "Z", (0,))
        for coefficient in (1e8, 1.0, -1e8)
    )
)

result = fq.experimental.numerics.parameter_shift_split_real_imag_precision_gradient(
    circuit,
    observable,
    parameter_bindings={"theta": 0.23},
    device="cpu",
)

print(result.gradient.high, result.gradient.low)
print(result.cpu_float64())  # CPU diagnostic reconstruction only
```

The accelerator result remains two FP32 words. `cpu_float64()` first transfers
those words to CPU, then reconstructs a diagnostic value; it is not part of
accelerator execution.

## Explicit contracts

The implemented plan is available from
`fq.experimental.numerics.split_real_imag_p2_precision_plan()`. Callers may pass its
machine-readable form back through `precision_plan=`. Any different plan fails
closed: P2 never silently upgrades the whole state, demotes a requested dtype,
or moves computation to CPU.

The bounded certified envelope is returned by
`fq.experimental.numerics.split_real_imag_p2_accuracy_envelope()`. A tighter requested
bound, state-infidelity or decomposition requirement, deterministic-execution
claim, or convergence-evidence requirement is rejected before state
allocation. Passing the envelope does not certify an arbitrary scientific
workload; it identifies the strongest checked-in P2 conformance boundary.

The authoritative machine contract is
[`split-real-imag-statevector-p2-precision-contract.toml`](../../contracts/split-real-imag-statevector-p2-precision-contract.toml),
and the packaged operator surface is
[`split_real_imag_statevector_p2_precision.json`](../../flagquantum/runtime/profiles/split_real_imag_statevector_p2_precision.json).

## Conformance

The P2 suite uses cancellation-sensitive Pauli Hamiltonians at depths 8, 32,
and 128 with two seeds. Its conformance workload places eight unit-coefficient
terms between coefficients `1e8` and `-1e8`; using several small terms avoids
an accidentally exact FP32 result caused by a backend-specific reduction
order. It compares selective Double-Single and ordinary P1 FP32 reductions
with a CPU complex128 reference. Both the absolute P2 error and its improvement
over FP32 must pass:

```bash
python -c 'import flagquantum as fq; r = fq.experimental.numerics.run_split_real_imag_precision_conformance("cpu"); r.require_accepted(); print(r.to_dict())'
```

In a Torch-FL CUDA-reference environment:

```bash
python tools/validate_split_real_imag_precision_flagos.py --device flagos:0
```

The 2026-08-24 NVIDIA A800 native CUDA and Torch-FL reference runs are recorded
in the checked-in
[`split_real_imag_precision_a800_20260824.json`](../../artifacts/split_real_imag_precision_a800_20260824.json)
artifact. CI validates its source revision, archive and profile identities,
full matrix, thresholds, route evidence, and non-promotion claims with
`tools/validate_split_real_imag_precision_evidence.py`.

CUDA and CUDA-backed `flagos:0` results are portability evidence only. They do
not certify a domestic accelerator or prove that provider-internal kernels
avoid every host-mediated route.

## Deliberate boundary

P2 is not complex128 emulation. FP32 state evolution error remains, and
Double-Single retains FP32 exponent range. Full Double-Single state storage,
Double-Single gate generation, native autograd, optimizer state, residual-word
checkpoints, decomposition, compilation, distributed execution, automatic
runtime selection, convergence, performance, and production use are outside
this phase. The ordinary CPU/CUDA statevector runtime and P0/P1 APIs are
unchanged, and Torch-FL remains an external validation dependency.
