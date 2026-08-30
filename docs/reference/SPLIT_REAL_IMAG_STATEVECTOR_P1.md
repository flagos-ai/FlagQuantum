# Split real/imag FP32 observable and parameter-shift P1

P1 extends the explicit split real/imag executor with bounded Pauli
expectation values and explicit parameter-shift gradients. State amplitudes,
gate matrices, observable reductions, and returned gradients remain ordinary
`torch.float32` tensors on the selected PyTorch device. No complex accelerator
tensor is constructed.

```python
import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, pauli_term

theta = fq.Parameter("theta")
phi = fq.Parameter("phi")
circuit = fq.Circuit(2).h(0).rx(0, theta=theta).ry(1, theta=phi).cx(0, 1)
observable = Hamiltonian(
    (
        pauli_term(0.7, "Z", (0,)),
        pauli_term(-0.4, "XX", (0, 1)),
    )
)
bindings = {"theta": 0.23, "phi": -0.31}

value = fq.experimental.execute_split_real_imag_expectation(
    circuit, observable, parameter_bindings=bindings, device="cpu"
)
gradient = fq.experimental.parameter_shift_split_real_imag_gradient(
    circuit, observable, parameter_bindings=bindings, device="cpu"
)

assert gradient.parameter_order == ("phi", "theta")
print(value.value, gradient.gradient)
```

`parameter_order` defines the correspondence between gradient entries and
parameter names. If one named parameter occurs in several supported gates,
P1 shifts every occurrence independently and sums its contributions. The
result also records `shifted_evaluations` so the execution cost is explicit.

## Contract and numerical evidence

The machine-readable boundary is
[`split-real-imag-statevector-p1-contract.toml`](../../contracts/split-real-imag-statevector-p1-contract.toml).
The packaged
[`split_real_imag_statevector_p1.json`](../../flagquantum/runtime/profiles/split_real_imag_statevector_p1.json)
profile requires the FP32 forward and backward operator surface used by the
training primitive. Preflight fails closed before the user workload is
allocated.

The conformance suite evaluates depths 8, 32, and 128 with two deterministic
seeds. It compares expectation values and parameter-shift gradients with an
independent CPU complex128 calculation and checks state norm drift:

```bash
python -c 'import flagquantum as fq; r = fq.experimental.run_split_real_imag_training_conformance("cpu"); r.require_accepted(); print(r.to_dict())'
```

In a Torch-FL CUDA-reference environment:

```bash
python tools/validate_split_real_imag_training_flagos.py --device flagos:0
```

The FlagOS report proves logical `flagos:0` residency and records that
FlagQuantum did not select a host fallback. It deliberately does not claim
that a provider's internal route avoided all host-mediated kernels, certify a
domestic accelerator, or establish performance or scalability.

The 2026-08-24 A800 native CUDA and Torch-FL reference runs are recorded in
[`split_real_imag_training_a800_20260824.json`](../../artifacts/split_real_imag_training_a800_20260824.json).
Validate the checked-in source, environment, route, coverage, numerical, and
claim boundaries without accelerator access:

```bash
python tools/validate_split_real_imag_training_evidence.py
```

## Gradient semantics

P1 supports direct named scalar parameters in `RX`, `RY`, `RZ`, `RXX`, `RYY`,
and `RZZ`. Its occurrence-wise two-term shift is an explicit numerical
gradient API, not PyTorch autograd. Callers own parameter updates and optimizer
state. Parameter expressions such as `2 * theta`, anonymous trainable tensors,
trainable Hamiltonian coefficients, and unsupported parameterized gates fail
closed instead of silently producing an approximate gradient.

Hamiltonians may use FlagQuantum `Hamiltonian`/`HamiltonianTerm` objects or IR
Pauli observable nodes. Batch size one and the built-in split gate set remain
the execution boundary. Custom initial states or matrices, sampling, native
autograd, optimizer integration, checkpointing, compilation, and distributed
execution are unsupported.

## Precision and runtime boundary

P1 is ordinary split FP32, not Double-Single and not an emulation of
complex128. Passing its bounded conformance thresholds is evidence for these
test circuits only; scientific convergence remains workload-specific. The API
is opt-in and does not change the ordinary statevector runtime, CUDA defaults,
or FlagQuantum's core installation dependencies. Torch-FL remains an external
platform provider imported only by its dedicated validator.
