"""Independent coin reflection and controlled cyclic permutation matrices."""

import torch

import flagquantum as fq

from ..physics import COIN_AMPLITUDES, POSITION_AMPLITUDES


def prepare(amplitudes: tuple[float, ...]) -> torch.Tensor:
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    identity = torch.eye(len(amplitudes), dtype=torch.complex128)
    delta = identity[:, 0] - vector
    norm = delta.vdot(delta)
    return (
        identity
        if abs(complex(norm)) < 1e-30
        else identity - 2 * torch.outer(delta, delta.conj()) / norm
    )


def shift_matrix(axis: int) -> torch.Tensor:
    matrix = torch.zeros((128, 128), dtype=torch.complex128)
    for coin in range(4):
        for position in range(32):
            offset = (1 if coin % 2 == 0 else -1) if coin // 2 == axis else 0
            matrix[32 * coin + (position + offset) % 32, 32 * coin + position] = 1
    return matrix


def reference_circuit(
    t: int = 2,
    x_amplitudes: tuple[float, ...] = POSITION_AMPLITUDES,
    y_amplitudes: tuple[float, ...] = POSITION_AMPLITUDES,
    coin_amplitudes: tuple[float, ...] = COIN_AMPLITUDES,
) -> fq.Circuit:
    circuit = fq.Circuit(12, dtype=torch.complex128)
    circuit.any(*range(5), matrix=prepare(x_amplitudes))
    circuit.any(*range(5, 10), matrix=prepare(y_amplitudes))
    circuit.any(10, 11, matrix=prepare(coin_amplitudes))
    reflection = torch.full((4, 4), 0.5, dtype=torch.complex128) - torch.eye(
        4, dtype=torch.complex128
    )
    for step in range(t):
        circuit.any(10, 11, matrix=reflection)
        circuit.any(10, 11, *range(5), matrix=shift_matrix(0))
        circuit.any(10, 11, *range(5, 10), matrix=shift_matrix(1))
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    expected = fq.run(reference_circuit()).statevector().flatten()
    actual = fq.run(program).statevector().reshape(4096, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
