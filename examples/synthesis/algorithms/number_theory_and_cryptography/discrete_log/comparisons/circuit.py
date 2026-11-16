"""Independent controlled modular permutations and dense Fourier matrices."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.number_theory_and_cryptography.discrete_log.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states


def reference_circuit(element: int = 3, generator: int = 7) -> fq.Circuit:
    circuit = fq.Circuit(14, dtype=torch.complex128).x(13)
    identity = torch.eye(16, dtype=torch.complex128)
    for offset, base in ((0, element), (5, generator)):
        for bit in range(5):
            wire = offset + 4 - bit
            multiplier = pow(base, 1 << bit, 13)
            block = torch.zeros((32, 32), dtype=torch.complex128)
            block[:16, :16] = identity
            for value in range(16):
                target = multiplier * value % 13 if value < 13 else value
                block[16 + target, 16 + value] = 1
            circuit.h(wire).any(wire, *range(10, 14), matrix=block)
    inverse_qft = torch.fft.fft(
        torch.eye(32, dtype=torch.complex128), dim=0, norm="ortho"
    )
    circuit.any(*range(5), matrix=inverse_qft)
    return circuit.any(*range(5, 10), matrix=inverse_qft)


def analytic_state(element: int = 3, generator: int = 7) -> torch.Tensor:
    state = torch.zeros((32, 32, 16), dtype=torch.complex128)
    for alpha in range(32):
        for beta in range(32):
            state[
                alpha, beta, pow(element, alpha, 13) * pow(generator, beta, 13) % 13
            ] = (1 / 32)
    transformed = torch.fft.fftn(state, dim=(0, 1), norm="ortho")
    assert isinstance(transformed, torch.Tensor)
    return transformed.flatten()


def run_example() -> None:
    reference = reference_circuit()
    actual = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual, analytic_state(), atol=1e-12, rtol=0)
    compare_states(build_program(), reference, ("alpha", "beta", "func_res"))


if __name__ == "__main__":
    run_example()
