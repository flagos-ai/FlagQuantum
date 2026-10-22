"""Train a moving pair's two pulse amplitudes on the existing CPU dynamics path.

Run ``python -m examples.train_rydberg_controls``. Geometry, C6 and loss are fixed;
only the two interval amplitude bindings are optimized. Detection is an effect,
not sampled events. Rebuild the live plan after each optimizer update.
"""

from __future__ import annotations

import argparse

import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.experimental.simulation import RydbergProgram


def moving_pair_program() -> RydbergProgram:
    program = RydbergProgram(
        atoms=(("a", "left"), ("b", "near")),
        sites={"left": (0.0, 0.0), "near": (2.0, 0.0), "far": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=64.0,
        loss_rate=0.1,
        minimum_separation=1.0,
        transport_channels=("aod",),
        drive_channels=("laser",),
    )
    program.move(
        "b", source="near", destination="far", start=0.0, duration=1.0, channel="aod"
    )
    program.pulse(
        ("a", "b"),
        start=0.0,
        duration=1.0,
        amplitude=fq.Parameter("omega"),
        detuning=0.2,
        phase=0.3,
        channel="laser",
    )
    return program


def control_plan(amplitudes: torch.Tensor, *, intervals: int = 128) -> fql.LindbladPlan:
    if type(intervals) is not int or intervals < 2 or intervals % 2:
        raise ValueError("intervals must be a positive even integer of at least two")
    if amplitudes.shape != (2,):
        raise ValueError("this example uses two scalar interval amplitudes")
    program = moving_pair_program()
    return fql.plan(
        program,
        program.basis_state(("g", "g")),
        torch.linspace(0.0, 1.0, intervals + 1, dtype=torch.float64),
        parameter_values=[
            {"omega": amplitudes[k // (intervals // 2)]} for k in range(intervals)
        ],
        outputs=(
            fq.expectation(program.projector({"a": "r", "b": "r"}), name="rr"),
            fq.expectation(program.survival_operator(), name="survival"),
            fq.expectation(program.no_flag_operator(0.8), name="no_flag"),
        ),
        method="runge-kutta",
        options=fq.ExecutionOptions(require_gradients=amplitudes.requires_grad),
    )


def train_controls(
    amplitudes: torch.Tensor, *, steps: int = 40, intervals: int = 128
) -> None:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    optimizer = torch.optim.Adam([amplitudes], lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        result = fql.run(control_plan(amplitudes, intervals=intervals))
        loss = (result.expectation("rr")[-1] - 0.12).square()
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--intervals", type=int, default=128)
    args = parser.parse_args()
    amplitudes = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)
    initial = fql.run(control_plan(amplitudes, intervals=args.intervals))
    rr = initial.expectation("rr")[-1]
    (gradient,) = torch.autograd.grad(rr, amplitudes)
    initial_loss = (rr.detach() - 0.12).square()
    print("Runtime: CPU, two actors, interval_midpoint")
    print(f"Initial unconditional RR: {rr.item():.8f}")
    print("Initial amplitude gradient:", [round(v, 8) for v in gradient.tolist()])
    train_controls(amplitudes, steps=args.steps, intervals=args.intervals)
    # Export a fresh detached plan for numeric replay, rather than serializing a graph.
    plan = control_plan(amplitudes.detach(), intervals=args.intervals)
    final = fql.run(plan)
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    rr = final.expectation("rr")[-1]
    survival, no_flag = (
        final.expectation(name)[-1] for name in ("survival", "no_flag")
    )
    if survival <= 0 or no_flag <= 0:
        raise RuntimeError("conditioning requires a positive denominator")
    print("Training reduced loss:", bool((rr - 0.12).square() < initial_loss / 100))
    print(f"Final unconditional RR: {rr.item():.8f}")
    print(f"Survival-conditioned RR: {(rr / survival).item():.8f}")
    print(f"No-flag-selected RR: {(rr / no_flag).item():.8f}")
    print("Exact numeric replay:", torch.equal(final.populations, replay.populations))


if __name__ == "__main__":
    main()
