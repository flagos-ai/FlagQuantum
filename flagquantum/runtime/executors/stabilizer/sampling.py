"""Plan-aware execution lifecycle for the Pauli-stabilizer representation.

The numerical work lives in `flagquantum/simulation/stabilizer`; this module owns
the Runtime side of the same route -- binding a plan's sampling request to the
engine, adapting the engine to the Runtime measurement contract, and keeping the
tableau's cost visible against a declared memory limit. Nothing here builds a
state, selects a seed stream, or decides a qubit order.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ....core.ir import CircuitIR
from ....errors import CapabilityError, ValidationError
from ...execution_plan import ExecutionPlan
from ...planner import plan_advanced as build_plan


class StabilizerSamplingTarget:
    """Adapts the Pauli-stabilizer engine to the measurement sampling contract.

    The Runtime measurement contract draws from any target exposing
    ``sample(shots, *, generator, format)``, which is what keeps postselection,
    qubit selection, counts, and shot statistics one implementation instead of one
    per state representation. This target answers for every qubit of the program in
    qubit order, so the contract's own qubit selection produces the requested
    columns, and it reads the seed the contract built from the request metadata
    rather than inventing a second seeding convention.

    Nothing is sampled at construction: the measurement contract decides how many
    drawings a request needs, and this target performs exactly one per call. The
    program it holds carries no lowered measurement nodes, because the qubits and
    shots are the request's, not the IR's. The returned tensor carries the leading
    batch axis the contract expects; a real batch would mean re-running one
    circuit rather than running a different one.
    """

    def __init__(self, program: CircuitIR, *, seed: int | None = None) -> None:
        self._program = program
        self._seed = seed

    @property
    def device(self) -> str:
        return "cpu"

    def sample(
        self,
        shots: int,
        *,
        generator: Any | None = None,
        format: str = "bits",
    ) -> torch.Tensor:
        from ....simulation.stabilizer import sample_stabilizer

        if format != "bits":
            raise CapabilityError(
                "stabilizer sampling returns bit strings; "
                f"format={format!r} has no stabilizer route"
            )
        return sample_stabilizer(
            self._program,
            shots=int(shots),
            seed=self._seed,
        ).unsqueeze(0)


def run_stabilizer_mode(
    execution_ir: CircuitIR,
    *,
    options: dict[str, Any],
    plan_options: dict[str, Any],
    provided_execution_plan: ExecutionPlan | None,
) -> tuple[Any, ExecutionPlan]:
    """Serve one sampling request through the Pauli-stabilizer engine.

    The planner has already refused a request this representation cannot answer,
    so this funnel only has to bind the plan's measurement node to the engine's
    entry point and keep the mode's cost visible: the table-sized ``state_bytes``
    is checked against a declared limit here, because a tableau has no smaller
    equivalent representation to fall back to.
    """

    requests = tuple(execution_ir.measurements)
    if len(requests) != 1:
        raise CapabilityError(
            "mode='stabilizer' serves exactly one sampling request per run, found "
            f"{len(requests)}; request it through fq.run(circuit, "
            "options=fq.ExecutionOptions(mode='stabilizer'), "
            "outputs=fq.samples(), shots=...)"
        )
    request = requests[0]
    if request.kind not in {"sample", "counts"}:
        raise CapabilityError(
            "mode='stabilizer' samples measurement outcomes; it cannot serve "
            f"measurement kind {request.kind!r}"
        )
    if request.shots is None:
        raise ValidationError("mode='stabilizer' requires a positive shot count")
    if request.metadata.get("postselect"):
        raise CapabilityError(
            "mode='stabilizer' cannot postselect: every drawing of one Clifford "
            "circuit is the same stream, so an unsatisfied condition would refill "
            "forever instead of converging"
        )
    execution_plan = provided_execution_plan or build_plan(
        execution_ir,
        state_mode="stabilizer",
        **plan_options,
    )
    limit = options.get("memory_limit_bytes")
    if limit is not None and execution_plan.state_bytes > int(limit):
        raise CapabilityError(
            "mode='stabilizer' needs "
            f"{execution_plan.state_bytes} bytes for the Clifford tableau, above "
            f"the declared memory_limit_bytes={int(limit)}; the tableau has no "
            "smaller equivalent representation to fall back to"
        )
    return (
        StabilizerSamplingTarget(
            replace(execution_ir, measurements=()),
            seed=request.metadata.get("seed"),
        ),
        execution_plan,
    )


__all__ = ("StabilizerSamplingTarget", "run_stabilizer_mode")
