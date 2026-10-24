"""Drive a weakly anharmonic three-level CPU model with four constant pulses.

The rotating-frame model has detuning 0.15, anharmonicity -3, and oscillator
ladder coupling sqrt(2) to level 2. Angular frequencies use inverse time units;
each segment lasts 0.25. No dissipation, device calibration or hardware pulse
emission is assumed. Refinement subdivides the same constant segments, not a
newly sampled waveform, so integration error is isolated from control error.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate


def pulse_plan(
    controls: torch.Tensor, *, subdivisions: int = 1, method: str = "krylov-exponential"
) -> fql.LindbladPlan:
    """Reuse interval bindings for the four in-phase/quadrature pulse pairs."""
    if controls.shape != (4, 2):
        raise ValueError("controls must have shape (4, 2)")
    if type(subdivisions) is not int or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    lower = annihilate(0).to_matrix({0: 3})
    x, y = fq.Parameter("x"), fq.Parameter("y")
    detuning = torch.diag(torch.tensor([0.0, 0.15, -2.7], dtype=torch.complex128))
    terms = [
        {"operator": detuning, "coefficient": 1.0},
        {"operator": lower + lower.mH, "coefficient": 0.5 * x},
        {"operator": 1j * (lower.mH - lower), "coefficient": 0.5 * y},
    ]
    return fql.plan(
        terms,
        [1.0, 0.0, 0.0],
        torch.arange(4 * subdivisions + 1, dtype=torch.float64) / (4 * subdivisions),
        dimensions=(3,),
        parameter_values=[
            {"x": row[0], "y": row[1]}
            for row in controls.unbind()
            for _ in range(subdivisions)
        ],
        outputs=[
            fq.expectation(torch.diag(row.to(torch.complex128)), name=name)
            for row, name in zip(
                torch.eye(3), ("ground", "excited", "leakage"), strict=True
            )
        ],
        method=method,
        return_density_matrices=True,
        options=fq.ExecutionOptions(require_gradients=controls.requires_grad),
    )


def reference_states(controls: torch.Tensor) -> torch.Tensor:
    """Independent 3x3 unitary products at the original pulse boundaries."""
    state = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    densities = [torch.outer(state, state.conj())]
    for x, y in controls.detach().tolist():
        drive = complex(x, -y) / 2
        hamiltonian = torch.tensor(
            [
                [0, drive, 0],
                [drive.conjugate(), 0.15, math.sqrt(2) * drive],
                [0, math.sqrt(2) * drive.conjugate(), -2.7],
            ],
            dtype=torch.complex128,
        )
        state = torch.matrix_exp(-0.25j * hamiltonian) @ state
        densities.append(torch.outer(state, state.conj()))
    return torch.stack(densities)


def initial_controls() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0], [0.6, 0.3], [-0.2, 0.7], [0.8, -0.1]], dtype=torch.float64
    )


def main() -> None:
    controls = initial_controls()
    reference = reference_states(controls)
    exact = fql.run(pulse_plan(controls))
    assert exact.density_matrices is not None
    torch.testing.assert_close(exact.density_matrices, reference, atol=1e-10, rtol=0)
    print(
        "Initial populations:",
        [
            round(float(exact.expectation(n)[-1]), 8)
            for n in ("ground", "excited", "leakage")
        ],
    )
    errors: list[float] = []
    for subdivisions in (1, 2, 4, 8):
        result = fql.run(
            pulse_plan(controls, subdivisions=subdivisions, method="runge-kutta")
        )
        assert result.density_matrices is not None
        error = float(abs(result.density_matrices[-1] - reference[-1]).max())
        errors.append(error)
        print(f"RK subdivisions {subdivisions}: density error={error:.2e}")
    assert all(a / b > 10 for a, b in pairwise(errors))
    fitted = torch.nn.Parameter(controls.clone())
    optimizer = torch.optim.Adam([fitted], lr=0.08)
    losses: list[float] = []
    for _ in range(40):
        optimizer.zero_grad()
        result = fql.run(pulse_plan(fitted))
        loss = (
            1 - result.expectation("excited")[-1]
        ).square() + 2 * result.expectation("leakage")[-1]
        torch.autograd.backward(loss)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] / 2
    numeric = pulse_plan(fitted.detach())
    trained = fql.run(numeric)
    replay = fql.run(fql.LindbladPlan.from_json(numeric.to_json()))
    assert trained.density_matrices is not None and replay.density_matrices is not None
    assert torch.equal(trained.density_matrices, replay.density_matrices)
    print(f"Training loss: {losses[0]:.2e} -> {losses[-1]:.2e}")
    print(
        "Trained excitation/leakage:",
        [round(float(trained.expectation(n)[-1]), 8) for n in ("excited", "leakage")],
    )
    print("Exact detached numeric replay: True")


if __name__ == "__main__":
    main()
