"""Fixed linear-path matrix reference for proposal 081, before native transport.

Two atoms follow prescribed positions (0, 0) and (2 + 4t, 0). Each local
space is ordered ground, Rydberg, lost. This is an effective model in consistent
arbitrary units, not a calibrated device or an implementation of RydbergProgram.
"""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql


def reference_plan(
    *,
    intervals: int = 128,
    method: str = "krylov-exponential",
    loss_rate: float = 0.2,
    efficiency: float = 0.8,
    batch_size: int | None = None,
) -> fql.LindbladPlan:
    """Freeze one fixed path at interval midpoints, using the current planner."""
    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 1:
        raise ValueError("intervals must be a positive integer")
    if not math.isfinite(loss_rate) or loss_rate < 0:
        raise ValueError("loss_rate must be finite and nonnegative")
    if not math.isfinite(efficiency) or not 0 <= efficiency <= 1:
        raise ValueError("efficiency must lie in [0, 1]")
    if batch_size not in (None, 1, 2):
        raise ValueError("this fixed reference supports an unbatched state or B=1/2")

    identity = torch.eye(3, dtype=torch.complex128)
    present = torch.diag(torch.tensor([1, 1, 0], dtype=torch.complex128))
    rydberg = torch.diag(torch.tensor([0, 1, 0], dtype=torch.complex128))
    no_flag = torch.diag(torch.tensor([1, 1, 1 - efficiency], dtype=torch.complex128))
    rr = torch.kron(rydberg, rydberg)
    coherence = torch.zeros((9, 9), dtype=torch.complex128)
    coherence[0, 4] = coherence[4, 0] = 1
    initial = torch.zeros(9, dtype=torch.complex128)
    initial[0] = initial[4] = 1 / math.sqrt(2)
    if batch_size is not None:
        initial = initial.unsqueeze(0).repeat(batch_size, 1)
        if batch_size == 2:
            initial[1] = 0
            initial[1, 8] = 1  # Absorbing lost/lost state cannot resurrect.
    jumps: list[dict[str, torch.Tensor | float]] = []
    for level in (0, 1):
        jump = torch.zeros((3, 3), dtype=torch.complex128)
        jump[2, level] = 1
        jumps.extend(
            {"operator": operator, "rate": loss_rate}
            for operator in (torch.kron(jump, identity), torch.kron(identity, jump))
        )
    times = torch.linspace(0.0, 0.5, intervals + 1, dtype=torch.float64)
    midpoint = (times[:-1] + times[1:]) / 2
    interaction = fq.Parameter("interaction")
    return fql.plan(
        [{"operator": rr, "coefficient": interaction}],
        initial,
        times,
        dimensions=(3, 3),
        parameter_values=[
            {"interaction": value} for value in 64 / (2 + 4 * midpoint) ** 6
        ],
        collapse_operators=jumps,
        outputs=(
            fq.expectation(torch.kron(present, present), name="survival"),
            fq.expectation(rr, name="rr"),
            fq.expectation(coherence, name="coherence"),
            fq.expectation(torch.kron(no_flag, no_flag), name="no_flag"),
        ),
        method=method,
        options=fq.ExecutionOptions(batch_size=batch_size),
        return_density_matrices=True,
    )


def main() -> None:
    plan = reference_plan()
    result = fql.run(plan)
    print(f"Physical survival: {result.expectation('survival')[-1].item():.8f}")
    print(f"Unconditional RR: {result.expectation('rr')[-1].item():.8f}")
    print(f"Linear-path coherence: {result.expectation('coherence')[-1].item():.8f}")
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    print("Exact numeric replay:", torch.equal(result.populations, replay.populations))
    print("Native transport API: pending implementation")


if __name__ == "__main__":
    main()
