"""Independent gate-level source walk comparison, including absolute phase."""

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.comparisons.circuit import (
    reference_circuit,
)

from ..workflow import run_workflow


def main() -> None:
    """Compare both candidates on the source input with the independent Circuit."""
    expected = fq.run(reference_circuit()).statevector().flatten()
    for candidate in run_workflow().candidates:
        logical = tuple(
            wire for name in ("x", "coin") for wire in candidate.program.registers[name]
        )
        width = candidate.resources.n_qubits
        hidden = tuple(wire for wire in range(width) if wire not in logical)
        vector = fq.run(candidate.conversion.program).statevector().flatten()
        ordered = (
            vector.reshape((2,) * width).permute((*logical, *hidden)).reshape(128, -1)
        )
        torch.testing.assert_close(ordered[:, 0], expected, atol=1e-11, rtol=0)
        assert float(ordered[:, 1:].abs().square().sum()) < 1e-24
        print(
            candidate.name,
            "maximum_state_error",
            float((ordered[:, 0] - expected).abs().max()),
        )


if __name__ == "__main__":
    main()
