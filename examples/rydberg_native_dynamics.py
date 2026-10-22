"""A moving atom pair on the existing CPU Lindblad and plan/replay path."""

from __future__ import annotations

import math

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.experimental.simulation import RydbergProgram


def native_pair_plan(
    *,
    intervals: int = 128,
    method: str = "krylov-exponential",
    loss_rate: float = 0.2,
    efficiency: float = 0.8,
    batch_size: int | None = None,
) -> fql.LindbladPlan:
    """Specify the source path once; the planner owns interval assembly."""
    program = RydbergProgram(
        atoms=(("a", "left"), ("b", "near")),
        sites={"left": (0.0, 0.0), "near": (2.0, 0.0), "far": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=64.0,
        loss_rate=loss_rate,
        minimum_separation=1.0,
        transport_channels=("aod",),
    )
    program.move(
        "b", source="near", destination="far", start=0.0, duration=0.5, channel="aod"
    )
    initial = (
        program.basis_state(("g", "g")) + program.basis_state(("r", "r"))
    ) / math.sqrt(2)
    if batch_size is not None:
        initial = initial.unsqueeze(0).repeat(batch_size, 1)
        if batch_size == 2:
            initial[1] = program.basis_state(("lost", "lost"))
    coherence = torch.zeros((9, 9), dtype=torch.complex128)
    coherence[0, 4] = coherence[4, 0] = 1
    return fql.plan(
        program,
        initial,
        torch.linspace(0.0, 0.5, intervals + 1, dtype=torch.float64),
        outputs=(
            fq.expectation(program.survival_operator(), name="survival"),
            fq.expectation(program.projector({"a": "r", "b": "r"}), name="rr"),
            fq.expectation(coherence, name="coherence"),
            fq.expectation(program.no_flag_operator(efficiency), name="no_flag"),
        ),
        method=method,
        options=fq.ExecutionOptions(batch_size=batch_size),
        return_density_matrices=True,
    )


def main() -> None:
    plan = native_pair_plan()
    result = fql.run(plan)
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    print("Native plan version:", plan.to_dict()["version"])
    print(f"Physical survival: {result.expectation('survival')[-1].item():.8f}")
    print(f"Unconditional RR: {result.expectation('rr')[-1].item():.8f}")
    print(f"No-flag probability: {result.expectation('no_flag')[-1].item():.8f}")
    print("Exact numeric replay:", torch.equal(result.populations, replay.populations))


if __name__ == "__main__":
    main()
