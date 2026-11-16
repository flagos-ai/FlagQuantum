"""Independent ideal channel and explicit measurement-branch reference."""

from math import pi

import torch

import flagquantum as fq


def reference_circuit(theta: float = pi / 2, phi: float = 0.0) -> fq.Circuit:
    return (
        fq.Circuit(3, dtype=torch.complex128)
        .ry(0, theta=theta)
        .phase(0, theta=phi)
        .h(1)
        .h(2)
    )


def measured_branches(input_state: torch.Tensor) -> torch.Tensor:
    """Return unnormalized corrected Bob/reference states for all four outcomes.

    Input axes are external reference and Alice. Output columns place Bob
    before the external reference. Enumerate the Bell-pair bit, Alice's H
    outcome, computational measurement and classical X/Z corrections explicitly.
    """
    reference_size, width = input_state.shape
    if width != 2:
        raise ValueError("Alice must have exactly two amplitudes")
    branches = torch.zeros((4, 2 * reference_size), dtype=torch.complex128)
    for reference in range(reference_size):
        for alice in range(2):
            for bell in range(2):
                for first in range(2):
                    second = alice ^ bell
                    amplitude = (
                        input_state[reference, alice] * (-1) ** (first * alice) / 2
                    )
                    corrected = bell ^ second
                    amplitude *= (-1) ** (first * corrected)
                    branches[
                        2 * first + second, corrected * reference_size + reference
                    ] += amplitude
    return branches


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    state = fq.run(program).statevector().reshape(8, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
