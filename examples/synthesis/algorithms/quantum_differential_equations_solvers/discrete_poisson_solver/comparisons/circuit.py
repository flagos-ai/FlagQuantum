"""Independent Fourier-derived transforms, analytic spectrum and inverse DFT."""

import math

import torch

import flagquantum as fq

from ..physics import AMPLITUDES, ANGLES


def spatial_transforms() -> tuple[torch.Tensor, torch.Tensor]:
    samples = torch.arange(8, dtype=torch.float64) + 0.5
    modes = torch.arange(9, dtype=torch.float64)
    fourier = torch.exp(1j * math.pi * torch.outer(modes, samples) / 8)
    cosine = fourier[:-1].real.clone() * 0.5
    sine = fourier[1:].imag.clone() * 0.5
    cosine[0] /= math.sqrt(2)
    sine[-1] /= math.sqrt(2)
    return sine.to(torch.complex128), cosine.to(torch.complex128)


def spectrum() -> torch.Tensor:
    # Observable indices count from the least significant register bit.
    # Retain the source's axis assignment rather than relabeling its operator.
    result = []
    for word in range(64):
        reversed_word = int(f"{word:06b}"[::-1], 2)
        x, y = reversed_word % 8, reversed_word // 8
        result.append(
            4 * math.sin(math.pi * (x + 1) / 16) ** 2
            + 4 * math.sin(math.pi * y / 16) ** 2
        )
    return torch.tensor(result, dtype=torch.float64)


def qpe_gates(circuit: fq.Circuit, inverse: bool = False) -> None:
    energies = spectrum()
    indices = torch.arange(64, dtype=torch.float64)
    fourier = torch.exp(2j * math.pi * torch.outer(indices, indices) / 64) / 8
    if inverse:
        circuit.any(*range(6, 12), matrix=fourier)
    for bit in reversed(range(6)) if inverse else range(6):
        duration = -2 * math.pi * 0.101321183642338 * (2**bit)
        diagonal = torch.exp((1j if inverse else -1j) * duration * energies)
        controlled = torch.eye(128, dtype=torch.complex128)
        controlled[64:, 64:] = torch.diag(diagonal)
        circuit.any(11 - bit, *range(6), matrix=controlled)
    if not inverse:
        circuit.any(*range(6, 12), matrix=fourier.mH)


def reference_circuit(amplitudes: tuple[float, ...] = tuple(AMPLITUDES)) -> fq.Circuit:
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    eye = torch.eye(64, dtype=torch.complex128)
    delta = eye[:, 0] - vector
    prep = (
        eye
        if abs(complex(delta.vdot(delta))) < 1e-24
        else eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    )
    circuit = fq.Circuit(13, dtype=torch.complex128).any(*range(6), matrix=prep)
    sine, cosine = spatial_transforms()
    circuit.any(3, 4, 5, matrix=sine).any(0, 1, 2, matrix=cosine)
    for wire in range(6, 12):
        circuit.h(wire)
    qpe_gates(circuit)
    rotations = torch.zeros((128, 128), dtype=torch.complex128)
    for index, angle in enumerate(ANGLES):
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        rotations[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.complex128
        )
    circuit.any(*range(6, 13), matrix=rotations)
    qpe_gates(circuit, inverse=True)
    for wire in range(6, 12):
        circuit.h(wire)
    return circuit.any(0, 1, 2, matrix=cosine.mH).any(3, 4, 5, matrix=sine.mH)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    assert program.report["workspace_qubits"] == 0
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
