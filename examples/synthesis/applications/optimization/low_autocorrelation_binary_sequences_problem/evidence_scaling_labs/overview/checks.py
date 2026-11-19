"""Check all named candidates against the independent LABS Circuit."""

import flagquantum as fq
from examples.synthesis.applications.optimization.low_autocorrelation_binary_sequences_problem.evidence_scaling_labs.comparisons.circuit import (
    reference_circuit,
)

from . import model, quantum

BINDINGS = (
    (0.013, -0.026, 0.039, -0.052, 0.065, -0.078),
    (-0.019, 0.046, -0.057, 0.092, -0.095, 0.138),
)


def verify(candidates: tuple[quantum.Candidate, ...]) -> tuple[float, ...]:
    errors = []
    for binding in BINDINGS:
        alternating = tuple(
            binding[index]
            for layer in range(model.LAYERS)
            for index in (layer, model.LAYERS + layer)
        )
        expected = fq.run(reference_circuit(alternating)).statevector().flatten()
        for candidate in candidates:
            physical = (
                fq.run(quantum.bind(candidate, binding))
                .statevector()
                .reshape((2,) * model.WIDTH)
            )
            actual = physical.permute(tuple(reversed(range(model.WIDTH)))).flatten()
            error = float((actual - expected).abs().max())
            if error >= 1e-11:
                raise AssertionError(
                    f"LABS candidate {candidate.schedule.value} absolute phase error {error}"
                )
            errors.append(error)
    return tuple(errors)
