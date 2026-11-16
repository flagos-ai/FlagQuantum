"""Compare Qmod and native qfunc against the independent Circuit."""

import math
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.bpde.main import (
    H0,
    H1,
    native_program,
)
from flagquantum.services.synthesis import _synthesize


def reference(delta: float) -> fq.Circuit:
    z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    for bit, (scalar, cz, cx) in enumerate((H0, H1)):
        half_z = torch.matrix_exp(-1j * 0.25 * cz * z)
        step = (half_z @ torch.matrix_exp(-1j * 0.5 * cx * x) @ half_z) * complex(
            math.cos(-0.5 * scalar), math.sin(-0.5 * scalar)
        )
        matrix[2 * bit : 2 * bit + 2, 2 * bit : 2 * bit + 2] = step.matrix_power(8)
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .x(1)
        .h(0)
        .any(0, 1, matrix=matrix)
        .phase(0, theta=4 * delta)
        .h(0)
    )


def run_example() -> None:
    source = Path(__file__).with_name("source.qmod")
    for delta in (0.37, -0.81):
        expected = fq.run(reference(delta)).statevector().reshape(-1)
        for program in (
            _synthesize(
                source,
                classical_inputs={"delta_e": delta},
                memory_limit_bytes=64 * 1024**2,
            ),
            native_program(delta),
        ):
            result = fq.run(program)
            actual = result.statevector().reshape(4, -1)
            torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
            assert float(actual[:, 1:].abs().square().sum()) < 1e-24
            print(delta, dict(program.probabilities(result, registers=("readout",))))
            print(program.report["resources"])
    print(
        "Original Qmod, native phase composition and Circuit agree, including identity-term interference."
    )


if __name__ == "__main__":
    run_example()
