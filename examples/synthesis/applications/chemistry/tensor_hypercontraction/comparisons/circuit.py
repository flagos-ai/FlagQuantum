"""Independent dense active-space Circuit without allocating the 83-bit state."""

import math

import torch

import flagquantum as fq

from ..main import CLOCK_AMPLITUDES


def reference_circuit(
    weights: tuple[float, ...] = CLOCK_AMPLITUDES,
    clock_phase: float = 0.0,
    spin_phase: float = 0.0,
) -> fq.Circuit:
    vector = torch.tensor(weights, dtype=torch.complex128)
    identity = torch.eye(16, dtype=torch.complex128)
    delta = identity[:, 0] - vector
    norm = delta.vdot(delta)
    prepare = (
        identity
        if abs(complex(norm)) < 1e-24
        else identity - 2 * torch.outer(delta, delta.conj()) / norm
    )
    indices = torch.arange(16, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(indices, indices) / 16) / 4
    circuit = fq.Circuit(6, dtype=torch.complex128).any(0, 1, 2, 3, matrix=prepare)
    for i in range(4):
        circuit.phase(3 - i, theta=clock_phase * (i + 1))
    return circuit.h(4).h(5).phase(5, theta=spin_phase).any(0, 1, 2, 3, matrix=inverse)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    result = fq.run(program)
    active = (*program.registers["t"], *program.registers["psi"][:2])
    words = []
    for index in range(64):
        bits = ["0"] * 83
        for wire, bit in zip(active, f"{index:06b}", strict=True):
            bits[wire] = bit
        words.append("".join(bits))
    state = result.native()
    observed = getattr(state, "amplitudes")(words)
    assert isinstance(observed, torch.Tensor)
    expected = fq.run(reference_circuit()).statevector()
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert float(observed.abs().square().sum()) > 1 - 1e-12
    assert not getattr(state, "truncation_records")
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
