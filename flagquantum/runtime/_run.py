"""Compose program planning and accepted-plan execution for local requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from ..errors import CapabilityError
from .execution_plan import ExecutionPlan

if TYPE_CHECKING:
    from flagquantum.core.noise import NoiseModel
    from flagquantum.runtime.results import ExecutionResult

    from ..circuit import Circuit
    from ..core.ir import CircuitIR, MeasurementNode
    from ..core.ising import IsingBraidProgram
    from .options import ExecutionOptions


def run(
    program_or_plan: Circuit | CircuitIR | ExecutionPlan | IsingBraidProgram,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: NoiseModel | None = None,
    initial_state: torch.Tensor | None = None,
    inputs: Mapping[str, float | torch.Tensor] | None = None,
) -> ExecutionResult:
    """Plan and execute a program, or execute one validated plan exactly."""

    from .plan_execution import execute_plan

    if inputs is not None and initial_state is not None:
        raise CapabilityError("inputs and initial_state cannot be combined yet")
    if isinstance(program_or_plan, ExecutionPlan):
        if options is not None:
            raise TypeError("options must be None when executing an ExecutionPlan")
        if measurements is not None:
            raise TypeError("measurements must be None when executing an ExecutionPlan")
        if noise_model is not None:
            raise TypeError("noise_model must be None when executing an ExecutionPlan")
        return execute_plan(program_or_plan, initial_state=initial_state, inputs=inputs)
    if inputs is not None:
        if noise_model is not None:
            raise CapabilityError("runtime inputs do not support noise models")
        from ._runtime_inputs import plan_runtime_inputs

        return execute_plan(
            plan_runtime_inputs(
                program_or_plan, inputs, options=options, measurements=measurements
            ),
            inputs=inputs,
        )
    if initial_state is not None:
        if noise_model is not None:
            raise CapabilityError(
                "external initial_state does not support noise models"
            )
        from ._initial_state import plan_initial_state

        return execute_plan(
            plan_initial_state(
                program_or_plan,
                initial_state,
                options=options,
                measurements=measurements,
            ),
            initial_state=initial_state,
        )
    from .planner import plan

    return execute_plan(
        plan(
            program_or_plan,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )
    )
