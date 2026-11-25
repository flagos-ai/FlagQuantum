"""Independent ungrouped Strang rotations and analytic inverse Fourier matrix."""

import math

import torch

import flagquantum as fq

from ..physics import STEPS, TERMS, ZERO_STATE, Term


def pauli_rotation(
    circuit: fq.Circuit, term: Term, angle: float, control_wire: int
) -> None:
    support, coefficient = term
    if not support:
        circuit.phase(control_wire, theta=-angle * coefficient)
        return
    wires = [5 - bit for bit, _ in support]
    for (bit, axis), wire in zip(support, wires, strict=True):
        if axis == "X":
            circuit.h(wire)
        elif axis == "Y":
            circuit.rx(wire, theta=math.pi / 2)
    for left, right in zip(wires, wires[1:], strict=False):
        circuit.cx(left, right)
    circuit.crz(control_wire, wires[-1], theta=2 * angle * coefficient)
    for left, right in reversed(list(zip(wires, wires[1:], strict=False))):
        circuit.cx(left, right)
    for (_, axis), wire in zip(support, wires, strict=True):
        if axis == "X":
            circuit.h(wire)
        elif axis == "Y":
            circuit.rx(wire, theta=-math.pi / 2)


def reference_circuit(
    scale: float = 1.0, amplitudes: tuple[float, ...] = ZERO_STATE
) -> fq.Circuit:
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    eye = torch.eye(64, dtype=torch.complex128)
    delta = eye[:, 0] - vector
    norm = delta.vdot(delta)
    preparation = (
        eye
        if abs(complex(norm)) < 1e-24
        else eye - 2 * torch.outer(delta, delta.conj()) / norm
    )
    circuit = fq.Circuit(13, dtype=torch.complex128).any(*range(6), matrix=preparation)
    for wire in range(6, 13):
        circuit.h(wire)
    for index, repetitions in enumerate(STEPS):
        angle = -2 * math.pi * scale * (2**index) / repetitions / 2
        for _ in range(repetitions):
            for term in TERMS:
                pauli_rotation(circuit, term, angle, 12 - index)
            for term in reversed(TERMS):
                pauli_rotation(circuit, term, angle, 12 - index)
    indices = torch.arange(128, dtype=torch.float64)
    inverse = torch.exp(
        -2j * math.pi * torch.outer(indices, indices) / 128
    ) / math.sqrt(128)
    return circuit.any(*range(6, 13), matrix=inverse)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
