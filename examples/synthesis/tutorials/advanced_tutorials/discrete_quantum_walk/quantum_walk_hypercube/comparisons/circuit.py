"""Independent dense hypercube permutation and uniform-coin reflection."""

import torch

import flagquantum as fq

from ..physics import COIN_AMPLITUDES, POSITION_AMPLITUDES


def reference_circuit(
    t: int = 2,
    position_amplitudes: tuple[float, ...] = POSITION_AMPLITUDES,
    coin_amplitudes: tuple[float, ...] = COIN_AMPLITUDES,
) -> fq.Circuit:
    position = torch.tensor(position_amplitudes, dtype=torch.complex128)
    coin = torch.tensor(coin_amplitudes, dtype=torch.complex128)
    position /= torch.linalg.vector_norm(position)
    coin /= torch.linalg.vector_norm(coin)
    vector = torch.kron(position, coin)
    identity = torch.eye(64, dtype=torch.complex128)
    difference = identity[:, 0] - vector
    norm = difference.vdot(difference)
    prepare = (
        identity
        if float(norm.real) < 1e-30
        else identity - 2 * torch.outer(difference, difference.conj()) / norm
    )
    diffuser = torch.full((4, 4), 0.5, dtype=torch.complex128) - torch.eye(
        4, dtype=torch.complex128
    )
    shift = torch.zeros((64, 64), dtype=torch.complex128)
    for x in range(16):
        for direction in range(4):
            shift[4 * (x ^ (1 << direction)) + direction, 4 * x + direction] = 1
    circuit = fq.Circuit(6, dtype=torch.complex128).any(*range(6), matrix=prepare)
    for step in range(t):
        circuit.any(4, 5, matrix=diffuser).any(*range(6), matrix=shift)
    return circuit


def run_example() -> None:
    from ..main import build_program

    for t in (0, 1, 2, 3):
        program = build_program(t)
        expected = fq.run(reference_circuit(t)).statevector().flatten()
        work = program.report["workspace_qubits"]
        if not isinstance(work, int):
            raise TypeError("Expected an integer workspace estimate")
        complete = torch.zeros((64, 1 << work), dtype=torch.complex128)
        complete[:, 0] = expected
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(),
            complete.flatten(),
            atol=1e-12,
            rtol=0,
        )
        print(t, program.report["resources"])


if __name__ == "__main__":
    run_example()
