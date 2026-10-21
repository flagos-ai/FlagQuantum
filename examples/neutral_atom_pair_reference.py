"""Current-API reference for two distance intervals and absorbing atom loss.

This is an effective matrix model, not native atom transport or a detector
trajectory. The two distances are piecewise constant; no physical path between
them is simulated. Length/time units are arbitrary but consistent, with angular
interaction coefficient C6=64 and hbar=1. No species or device is calibrated.
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql


def reference_plan(
    *,
    distances: tuple[float, float] = (2.0, 4.0),
    loss_rate: float = 0.2,
    detection_efficiency: float = 0.8,
) -> fql.LindbladPlan:
    """Build a two-interval reference on local states ground/Rydberg/lost."""
    if not all(math.isfinite(r) and r > 0 for r in distances):
        raise ValueError("reference distances must be positive and finite")
    if not math.isfinite(loss_rate) or loss_rate < 0:
        raise ValueError("reference loss rate must be nonnegative and finite")
    if not math.isfinite(detection_efficiency) or not 0 <= detection_efficiency <= 1:
        raise ValueError("reference detection efficiency must lie in [0, 1]")

    identity = torch.eye(3, dtype=torch.complex128)
    present = torch.diag(torch.tensor([1, 1, 0], dtype=torch.complex128))
    rydberg = torch.diag(torch.tensor([0, 1, 0], dtype=torch.complex128))
    rr = torch.kron(rydberg, rydberg)
    no_flag = torch.diag(
        torch.tensor([1, 1, 1 - detection_efficiency], dtype=torch.complex128)
    )
    coherence = torch.zeros((9, 9), dtype=torch.complex128)
    coherence[0, 4] = coherence[4, 0] = 1
    jumps: list[dict[str, torch.Tensor | float]] = []
    for level in (0, 1):
        jump = torch.zeros((3, 3), dtype=torch.complex128)
        jump[2, level] = 1
        jumps.extend(
            {"operator": operator, "rate": loss_rate}
            for operator in (torch.kron(jump, identity), torch.kron(identity, jump))
        )
    initial = torch.zeros(9, dtype=torch.complex128)
    initial[0] = initial[4] = 1 / math.sqrt(2)
    interaction = fq.Parameter("interaction")
    return fql.plan(
        [{"operator": rr, "coefficient": interaction}],
        initial,
        [0.0, 0.25, 0.5],
        dimensions=(3, 3),
        parameter_values=[{"interaction": 64 / r**6} for r in distances],
        collapse_operators=jumps,
        method="krylov-exponential",
        outputs=(
            fq.expectation(torch.kron(present, present), name="survival"),
            fq.expectation(rr, name="rr"),
            fq.expectation(coherence, name="coherence"),
            fq.expectation(torch.kron(no_flag, no_flag), name="no_flag"),
        ),
    )


def main() -> None:
    plan = reference_plan()
    result = fql.run(plan)
    survival = result.expectation("survival")[-1]
    rr = result.expectation("rr")[-1]
    no_flag = result.expectation("no_flag")[-1]
    print(f"Physical survival: {survival.item():.8f}")
    print(f"Unconditional RR: {rr.item():.8f}")
    print(f"Survival-conditioned RR: {(rr / survival).item():.8f}")
    print(f"No-flag-selected RR: {(rr / no_flag).item():.8f}")
    print(f"Detected-loss probability: {(1 - no_flag).item():.8f}")
    print(
        f"Distance-dependent coherence: {result.expectation('coherence')[-1].item():.8f}"
    )
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    print(
        "Exact numeric replay:",
        torch.equal(replay.expectation("rr"), result.expectation("rr")),
    )


if __name__ == "__main__":
    main()
