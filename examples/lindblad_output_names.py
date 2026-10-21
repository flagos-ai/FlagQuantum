"""Plan named Lindblad outputs and refuse silent name collisions.

Run with ``python -m examples.lindblad_output_names`` on CPU.
"""

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.simulation.lindblad import EvolutionValidationError


def main() -> None:
    hamiltonian = torch.zeros((2, 2), dtype=torch.complex128)
    outputs = [fq.expectation(fq.Z(0), name="polarization"), fq.expectation(fq.X(0))]
    result = fql.run(hamiltonian, "0", [0.0, 0.1], outputs=outputs)
    print(f"Polarization: {result.expectation('polarization')[-1].item():.1f}")
    print(f"X expectation: {result.expectation('expectation_1')[-1].item():.1f}")
    try:
        fql.plan(
            hamiltonian,
            "0",
            [0.0, 0.1],
            outputs=[
                fq.expectation(fq.Z(0), name="expectation_1"),
                fq.expectation(fq.X(0)),
            ],
        )
    except EvolutionValidationError as error:
        print(f"Collision refused: {error.code}")


if __name__ == "__main__":
    main()
