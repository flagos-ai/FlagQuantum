"""Independent source Ising basis-energy phases and explicit Circuit RX layers."""

import torch

import flagquantum as fq
from examples.synthesis.cost_reference import cost_phase_circuit


def source_energy(bits: tuple[int, ...]) -> float:
    """Sum source edge energies with half-field contributions at both endpoints."""
    spins = tuple(2 * bit - 1 for bit in bits)
    return sum(
        -10 * spins[i] * spins[(i + 1) % 6] + 10 * (spins[i] + spins[(i + 1) % 6])
        for i in range(6)
    )


def reference_circuit(parameters: tuple[float, ...]) -> fq.Circuit:
    return cost_phase_circuit(source_energy, 6, parameters)


def run_example() -> None:
    from ..workflow import optimize, source_program

    result = optimize()
    for parameters in (
        (0.011, 0.017, 0.023, 0.029, 0.031, 0.13, 0.19, 0.23, 0.29, 0.31),
        (-0.013, 0.021, -0.034, 0.015, -0.026, 0.16, -0.19, 0.32, -0.25, 0.34),
        result.parameters,
    ):
        program = source_program(parameters)
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(parameters)).statevector().flatten()
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        assert program.report["workspace_qubits"] == 0
        print("Maximum absolute state error:", float((actual - expected).abs().max()))
    print("Source-energy states, absolute phase, bit order and zero work agree.")


if __name__ == "__main__":
    run_example()
