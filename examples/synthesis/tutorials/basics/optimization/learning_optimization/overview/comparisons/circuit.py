"""Independent four-qubit reference using a classical cost diagonal."""

import cmath

import torch

import flagquantum as fq


def reference_circuit(gamma: float, beta: float) -> fq.Circuit:
    """Use MSB order x then y, explicit sixteen costs, and bitwise mixing.

    The 16-by-16 complex128 reference matrix uses 4096 bytes. It is only an
    independent correctness reference; synthesis does not consume this matrix.
    """
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for bit in range(4):
        circuit.h(bit)
    phases = []
    for word in range(16):
        x, y = divmod(word, 4)
        violation = max(0, 2 - 3 * x - y)
        cost = 3 * x + 2 * y + 20 * violation**2
        phases.append(cmath.exp(-1j * gamma * cost))
    circuit.any(
        0,
        1,
        2,
        3,
        matrix=torch.diag(torch.tensor(phases, dtype=torch.complex128)),
    )
    for bit in range(4):
        circuit.rx(bit, 2 * beta)
    return circuit


def main() -> None:
    """Display the reference named-assignment probabilities at nonzero angles."""
    state = fq.run(reference_circuit(0.19, 0.37)).statevector().flatten()
    print({divmod(word, 4): float(p) for word, p in enumerate(state.abs().square())})


if __name__ == "__main__":
    main()
