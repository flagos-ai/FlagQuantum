"""Train four qutrit pulses with fixed 2->1->0 relaxation on CPU.

Rates 0.4 and 0.7 use inverse time units, independently of angular frequencies.
This finite rotating-frame model is not calibrated hardware or a gate-fidelity
estimate. The closed-model example and its independent unitary oracle are intact.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.driven_three_level import initial_controls
from flagquantum.observables.boson import annihilate


def relaxation_plan(
    controls: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    rates: tuple[float, float] = (0.4, 0.7),
    subdivisions: int = 1,
    method: str = "krylov-exponential",
    undriven: bool = False,
) -> fql.LindbladPlan:
    """Compose existing numeric planning, fixed jumps and interval bindings.

    Examples:
        >>> result = fql.run(relaxation_plan(initial_controls()))
        >>> result.expectation("excited").shape
        torch.Size([5])
    """
    if controls.shape != (4, 2):
        raise ValueError("controls must have shape (4, 2)")
    if type(subdivisions) is not int or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    lower = annihilate(0).to_matrix({0: 3})
    x, y = fq.Parameter("x"), fq.Parameter("y")
    detuning = torch.diag(torch.tensor([0.0, 0.15, -2.7], dtype=torch.complex128))
    if undriven:
        terms = [{"operator": torch.zeros_like(lower), "coefficient": 1.0}]
    else:
        terms = [
            {"operator": detuning, "coefficient": 1.0},
            {"operator": lower + lower.mH, "coefficient": 0.5 * x},
            {"operator": 1j * (lower.mH - lower), "coefficient": 0.5 * y},
        ]
    jumps = []
    for source, target, rate in ((1, 0, rates[0]), (2, 1, rates[1])):
        jump = torch.zeros((3, 3), dtype=torch.complex128)
        jump[target, source] = 1
        jumps.append({"operator": jump, "rate": rate})
    bindings: list[Mapping[str | fq.Parameter, torch.Tensor]] | None = (
        None
        if undriven
        else [
            {"x": row[0], "y": row[1]}
            for row in controls.unbind()
            for _ in range(subdivisions)
        ]
    )
    return fql.plan(
        terms,
        (
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
            if initial is None
            else initial
        ),
        torch.arange(4 * subdivisions + 1, dtype=torch.float64) / (4 * subdivisions),
        dimensions=(3,),
        collapse_operators=jumps,
        parameter_values=bindings,
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


def main() -> None:
    controls = torch.nn.Parameter(initial_controls())
    optimizer = torch.optim.Adam([controls], lr=0.08)
    initial_loss = 0.0
    for step in range(40):
        optimizer.zero_grad()
        result = fql.run(relaxation_plan(controls))
        loss = (
            1 - result.expectation("excited")[-1]
        ).square() + 2 * result.expectation("leakage")[-1]
        if step == 0:
            initial_loss = float(loss.detach())
        torch.autograd.backward(loss)
        optimizer.step()
    plan = relaxation_plan(controls.detach())
    result = fql.run(plan)
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    final_loss = float(
        (1 - result.expectation("excited")[-1]).square()
        + 2 * result.expectation("leakage")[-1]
    )
    assert final_loss <= 0.9 * initial_loss
    assert result.density_matrices is not None and replay.density_matrices is not None
    assert torch.equal(result.density_matrices, replay.density_matrices)
    print("Fixed relaxation rates: gamma_10=0.4, gamma_21=0.7")
    print(f"Training objective: {initial_loss:.8f} -> {final_loss:.8f}")
    print(
        "Final ground/excited/leakage:",
        [
            round(float(result.expectation(n)[-1]), 8)
            for n in ("ground", "excited", "leakage")
        ],
    )
    print("Exact detached numeric replay: True")


if __name__ == "__main__":
    main()
