"""Four-mode fermionic Fourier measurement and inverse in JW occupation order.

Run: python -m examples.algorithms.fermion_fourier
"""

import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.lowering import fermionic_fourier_transform
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.observables import number


def main() -> None:
    print(
        "premise: exact CPU statevector in JW occupation order; "
        "adjacent routing has quadratic gate cost; no sampling or hardware noise.\n"
    )
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.x(1)
    initial = circuit.state().clone()
    circuit.compose(fermionic_fourier_transform(4))
    for mode in range(4):
        encoded = encode_fermion(number(mode), n_modes=4, encoding="jordan_wigner")
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        value = float(torch.as_tensor(result.expectation()).reshape(-1)[0])
        print(f"Mode {mode} occupation: {value:.6f}")
    circuit.compose(fermionic_fourier_transform(4, inverse=True))
    residual = float((circuit.state() - initial).abs().max())
    print(f"Inverse amplitude residual: {residual:.3e}")


if __name__ == "__main__":
    main()
