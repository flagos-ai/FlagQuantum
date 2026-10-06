"""Plan-aware execution lifecycle for the Pauli-stabilizer representation.

The numerical work lives in `flagquantum/simulation/stabilizer`; this module owns
the Runtime side of the same route -- binding a plan's measurement requests to the
engine, adapting the representation to the Runtime measurement contract, and
keeping the tableau's cost visible against a declared memory limit. Nothing here
builds a state, selects a seed stream, or decides a wire order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from ....core.ir import CircuitIR
from ....errors import CapabilityError, ValidationError
from ...execution_plan import ExecutionPlan
from ...planner import plan_advanced as build_plan


class StabilizerTarget:
    """Adapts the Pauli-stabilizer engine to the Runtime measurement contract.

    The contract draws from any target exposing ``sample(shots, *, generator,
    format)`` and reads a Pauli expectation from any target exposing
    ``expectation_ps`` and ``expectation_z``, which is what keeps postselection,
    wire selection, counts, shot statistics, the analytic readout, the observable
    flip, and the coefficient weighting one implementation instead of one per
    state representation. This target answers for every wire of the program in
    wire order, so the contract's own wire selection produces the requested
    columns, and it reads the seed the contract built from the request metadata
    rather than inventing a second seeding convention.

    Nothing is sampled at construction: the measurement contract decides how many
    drawings a request needs, and this target performs exactly one per call. A
    readout is not sampled at all, because a Clifford circuit expanded through a
    Pauli string is one Pauli and one exact value. The program it holds carries no
    lowered measurement nodes, because the wires and shots are the request's, not
    the IR's. The returned tensors carry the leading batch axis the contract
    expects; a real batch would mean re-running one circuit rather than running a
    different one.

    A program that carries channels is sampled through the engine's positioned
    noise entry point and a clean one through its noiseless entry point. Which of
    the two applies is read from the program rather than passed alongside it, so
    the route cannot hold a program and a claim about it that disagree; the engine
    owns that classification and the target asks it.
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
        from ....simulation.stabilizer import (
            sample_noisy_measurements,
            sample_stabilizer,
            survey_stabilizer_program,
        )

        if format != "bits":
            raise CapabilityError(
                "stabilizer sampling returns bit strings; "
                f"format={format!r} has no stabilizer route"
            )
        # The engine's own census decides which of its two sampling entry points
        # applies, because a channel reaches the IR both as a schema-declared
        # opcode and as a lowered instruction and only the engine reads both. All
        # wires are terminal here: the contract selects the columns it asked for,
        # so this target answers for the whole register in wire order either way.
        if survey_stabilizer_program(self._program).noise_instructions:
            samples = sample_noisy_measurements(
                self._program,
                shots=int(shots),
                terminal_wires=tuple(range(self._program.n_wires)),
                seed=self._seed,
            )
        else:
            samples = sample_stabilizer(
                self._program,
                shots=int(shots),
                seed=self._seed,
            )
        return samples.unsqueeze(0)

    def expectation_ps(
        self,
        *,
        z: Sequence[int] | None = None,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return the exact expectation of one Pauli string, as one number."""

        return self._readout(x=x, y=y, z=z).reshape(1)

    def expectation_z(
        self,
        qubits: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Return one exact `Z` expectation per requested qubit, in that order."""

        # The name is a wire-qualified parameter on a public method, so it takes
        # the spelling the qubit-vocabulary migration is retiring `wires` toward;
        # the measurement contract calls this positionally, and a caller reaching
        # it by keyword is asking the target directly rather than through
        # Runtime. A `bool` lands in the single-qubit branch the way `int` does,
        # and the engine's own letter lookup is what refuses it by name.
        if qubits is None:
            selected: tuple[int, ...] = tuple(range(self._program.n_wires))
        elif isinstance(qubits, int):
            selected = (qubits,)
        else:
            selected = tuple(qubits)
        return torch.cat(
            [self._readout(z=(qubit,)).reshape(1) for qubit in selected], dim=0
        ).reshape(1, -1)

    def _readout(
        self,
        *,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
        z: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return the program's exact value for one Pauli string."""

        from ....simulation.stabilizer import pauli_readout

        # Core refuses any dtype name outside `complex64` and `complex128` before
        # an IR exists, so this is a total map and not a validated one: one circuit
        # answers in the real dtype of the pair, whichever representation served
        # it. The sibling statevector targets read a `torch.dtype` the same way.
        dtype = torch.float32 if self._program.dtype == "complex64" else torch.float64
        readout = pauli_readout(self._program, x=x or (), y=y or (), z=z or ())
        return torch.tensor(float(readout.expectation), dtype=dtype)


def run_stabilizer_mode(
    execution_ir: CircuitIR,
    *,
    options: dict[str, Any],
    plan_options: dict[str, Any],
    provided_execution_plan: ExecutionPlan | None,
) -> tuple[Any, ExecutionPlan]:
    """Serve one run's measurement requests through the Pauli-stabilizer engine.

    The planner has already refused a request this representation cannot answer,
    so this funnel only has to bind the plan's measurement nodes to the engine's
    entry points and keep the mode's cost visible: the table-sized ``state_bytes``
    is checked against a declared limit here, because a tableau has no smaller
    equivalent representation to fall back to.

    A run may carry several readouts and at most one sampling request. The seed is
    read from the sampling request rather than from the first request, because the
    first request is a readout as soon as a run mixes the two, and a readout
    carries no seed to give the sampler.

    A scene-level model was already lowered into positioned channel instructions
    by the dispatcher that called this funnel, the way the density-matrix route
    lowers it, so an inline channel and an equivalent model arrive here as the same
    program. This module holds no Compiler dependency of its own, and the program
    it receives is the one it samples.
    """

    from ....simulation.stabilizer import (
        STABILIZER_MEASUREMENT_KINDS,
        STABILIZER_SAMPLING_KINDS,
    )

    requests = tuple(execution_ir.measurements)
    if not requests:
        raise CapabilityError(
            "mode='stabilizer' serves measurement requests, and this plan carries "
            "none; request samples with outputs=fq.samples() and a shot count, or "
            "an expectation with outputs=fq.expectation(...)"
        )
    unsupported = sorted(
        {request.kind for request in requests} - STABILIZER_MEASUREMENT_KINDS
    )
    if unsupported:
        raise CapabilityError(
            "mode='stabilizer' samples measurement outcomes and reads exact Pauli "
            "expectations; it cannot serve measurement kind(s) "
            f"{', '.join(unsupported)}"
        )
    sampling = tuple(
        request for request in requests if request.kind in STABILIZER_SAMPLING_KINDS
    )
    if len(sampling) > 1:
        raise CapabilityError(
            "mode='stabilizer' serves exactly one sampling request per run, found "
            f"{len(sampling)}; request it through fq.run(circuit, "
            "options=fq.ExecutionOptions(mode='stabilizer'), "
            "outputs=fq.samples(), shots=...)"
        )
    for request in requests:
        if request.metadata.get("postselect"):
            raise CapabilityError(
                "mode='stabilizer' cannot postselect: every drawing of one Clifford "
                "circuit is the same stream, so an unsatisfied condition would refill "
                "forever instead of converging"
            )
    if sampling and sampling[0].shots is None:
        raise ValidationError("mode='stabilizer' requires a positive shot count")
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
        StabilizerTarget(
            replace(execution_ir, measurements=()),
            seed=sampling[0].metadata.get("seed") if sampling else None,
        ),
        execution_plan,
    )


__all__ = ("StabilizerTarget", "run_stabilizer_mode")
