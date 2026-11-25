"""Independent classical stream/reflection permutations in a twelve-bit Circuit."""

from functools import lru_cache

import torch

import flagquantum as fq

from ..physics import ACTIVE_VELOCITIES, VELOCITY_AMPLITUDES, X_AMPLITUDES, Y_AMPLITUDES


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


@lru_cache(maxsize=3)
def stream_matrix(active: int) -> torch.Tensor:
    matrix = torch.zeros((64, 64), dtype=torch.complex128)
    for u in range(4):
        for direction in range(2):
            for position in range(8):
                offset = (1 if direction else -1) if u == active else 0
                before = (u * 2 + direction) * 8 + position
                after = (u * 2 + direction) * 8 + (position + offset) % 8
                matrix[after, before] = 1
    return matrix


@lru_cache(maxsize=6)
def reflection_matrix(active: int, axis: int) -> torch.Tensor:
    matrix = torch.zeros((512, 512), dtype=torch.complex128)
    edges, lower, upper = ((1, 5), 2, 5) if axis == 1 else ((2, 5), 1, 5)
    for fixed in range(8):
        for change in range(8):
            for u in range(4):
                hit = change in edges and lower <= fixed <= upper and u == active
                for direction in range(2):
                    prefix = ((fixed * 8 + change) * 4 + u) * 2
                    matrix[prefix + (direction ^ hit), prefix + direction] = 1
    return matrix


def reference_circuit(
    x_amplitudes: tuple[float, ...] = X_AMPLITUDES,
    y_amplitudes: tuple[float, ...] = Y_AMPLITUDES,
    ux_amplitudes: tuple[float, ...] = VELOCITY_AMPLITUDES,
    uy_amplitudes: tuple[float, ...] = VELOCITY_AMPLITUDES,
) -> fq.Circuit:
    circuit = fq.Circuit(12, dtype=torch.complex128)
    circuit.any(0, 1, 2, matrix=prepare(x_amplitudes))
    circuit.any(3, 4, 5, matrix=prepare(y_amplitudes))
    circuit.any(8, 9, matrix=prepare(ux_amplitudes))
    circuit.any(10, 11, matrix=prepare(uy_amplitudes))
    circuit.x(6).h(7)
    for active in ACTIVE_VELOCITIES:
        circuit.any(8, 9, 6, 0, 1, 2, matrix=stream_matrix(active))
        circuit.any(10, 11, 7, 3, 4, 5, matrix=stream_matrix(active))
        circuit.any(0, 1, 2, 3, 4, 5, 10, 11, 7, matrix=reflection_matrix(active, 1))
        circuit.any(3, 4, 5, 0, 1, 2, 8, 9, 6, matrix=reflection_matrix(active, 0))
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
