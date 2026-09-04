"""Exact execution path for a validated executable plan."""

from __future__ import annotations

import torch

from ..compilation.execution_plan_contract import (
    ExecutionPlanContractError,
    plan_decision,
    plan_execution_program,
    plan_noise_model,
    plan_program,
    validate_plan_environment,
)
from ..compilation.models import ExecutionPlan
from ..core.runtime_config import RuntimeConfig
from ..errors import ExecutionError, FlagQuantumError
from .measurements import validate_measurements
from .result import ExecutionResult


def execute_plan(execution_plan: ExecutionPlan) -> ExecutionResult:
    """Execute one verified plan without invoking planner or compiler again."""

    from .execution import _normalize_execution_output, run_native

    validate_plan_environment(execution_plan)
    source_ir = plan_program(execution_plan)
    execution_ir = plan_execution_program(execution_plan)
    noise_model = plan_noise_model(execution_plan)
    decision = plan_decision(execution_plan)
    requests = tuple(source_ir.measurements)
    validate_measurements(requests, n_wires=source_ir.n_wires)
    mode = str(decision["mode"])
    if int(decision["world_size"]) > 1:
        mode = {
            "statevector": "distributed_statevector",
            "mps": "distributed_mps",
            "tensor_network": "distributed_tensor_network",
        }.get(mode, mode)
    targets = {
        "auto": "full_state",
        "state": "full_state",
        "expectation": "expectation",
        "samples": "samples",
        "amplitudes": "few_amplitudes",
    }
    precision = str(decision["precision"])
    selected_config = RuntimeConfig(
        backend=str(decision["backend"]),
        device=str(decision["device"]),
        complex_dtype=precision,
        real_dtype="float64" if precision == "complex128" else "float32",
        jax_enable_x64=precision == "complex128",
    )
    try:
        output, returned_plan = run_native(
            execution_ir,
            noise_model=noise_model,
            mode=mode,
            return_plan=True,
            _execution_plan=execution_plan,
            bsz=int(decision["batch_size"]),
            world_size=int(decision["world_size"]),
            device=str(decision["device"]),
            dtype=getattr(torch, precision),
            config=selected_config,
            memory_limit_bytes=decision["memory_limit_bytes"],
            output_target=targets[str(decision["target"])],
            require_gradients=bool(decision["require_gradients"]),
            allow_approximate=bool(decision["allow_approximate"]),
        )
    except FlagQuantumError:
        raise
    except Exception as error:
        raise ExecutionError("planned execution failed") from error
    if returned_plan is not execution_plan:
        raise ExecutionPlanContractError(
            "identity_mismatch",
            "executor replaced the supplied plan instead of executing it",
        )
    return _normalize_execution_output(
        output,
        execution_plan,
        source_ir=source_ir,
        requests=requests,
        mode=mode,
        noise_model=noise_model,
    )


__all__ = ("execute_plan",)
