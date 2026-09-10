"""Private Runtime handoff for compiler-lowered measurement-feedback sessions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from ...core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ...core.parameters import is_parameterized_value
from ...noise import NoiseModel
from ._conditions import instruction_condition_clauses
from ._feedback import DynamicFeedbackPlan
from .circuit import DynamicCircuit
from .execution import run_dynamic
from .result import DynamicExecutionResult

_ALLOWED_FIXED_GATES = {"h", "x", "cx"}
_ALLOWED_ROTATIONS = {"rx", "ry"}
_SESSION_METADATA = {
    "hybrid_dynamic_session",
    "hybrid_dynamic_return_bit",
    "hybrid_dynamic_measurement_count",
    "hybrid_stochastic_gradient_policy",
    "hybrid_parameter_order",
    "hybrid_dynamic_unrolled_iterations",
}


def _contains_trainable(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, Mapping):
        return any(_contains_trainable(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_trainable(item) for item in value)
    return False


def _validate_session_header(ir: CircuitIR) -> None:
    if ir.metadata.get("hybrid_dynamic_session") is not True:
        raise ValueError("CircuitIR is not a compiler-lowered hybrid dynamic session")
    if set(ir.metadata) != _SESSION_METADATA:
        raise ValueError(
            "hybrid dynamic session metadata is outside the Phase 7 profile"
        )
    if ir.dtype not in {"complex64", "complex128"}:
        raise ValueError("hybrid dynamic session requires complex64 or complex128")
    if ir.observables or ir.measurements:
        raise ValueError(
            "hybrid dynamic session returns trajectory data, not static measurements"
        )


def _validate_bound_instruction(instruction: Instruction) -> None:
    if _contains_trainable(instruction.params) or _contains_trainable(
        instruction.matrix
    ):
        raise RuntimeError(
            "hybrid dynamic session stochastic gradients are unsupported"
        )
    if is_parameterized_value(instruction.params) or is_parameterized_value(
        instruction.matrix
    ):
        raise ValueError("hybrid dynamic CircuitIR must be bound before execution")


def _validated_condition_clauses(
    instruction: Instruction,
    instruction_index: int,
    measured: set[int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    clauses: tuple[tuple[tuple[int, int], ...], ...] = instruction_condition_clauses(
        instruction
    )
    if any(
        bit < 0 or expected not in {0, 1}
        for clause in clauses
        for bit, expected in clause
    ) or any(len({bit for bit, _ in clause}) != len(clause) for clause in clauses):
        raise ValueError(
            "classical condition clauses require unique bits and binary values"
        )
    if any(tuple(sorted(clause)) != clause for clause in clauses):
        raise ValueError("classical condition clauses must be canonically ordered")
    if "condition_clauses" in instruction.metadata and (
        len(clauses) < 2
        or tuple(sorted(clauses, key=lambda item: (len(item), item))) != clauses
        or len(set(clauses)) != len(clauses)
        or any(
            set(left).issubset(right)
            for clause_index, left in enumerate(clauses)
            for right in clauses[clause_index + 1 :]
        )
    ):
        raise ValueError("condition_clauses must be canonical, distinct, and minimal")
    if any(bit not in measured for clause in clauses for bit, _ in clause):
        raise ValueError(
            f"instruction {instruction_index} reads a classical bit before measurement"
        )
    return clauses


def _validate_session_ir(circuit_or_ir: Any) -> tuple[CircuitIR, int]:
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_session_header(ir)
    measured: set[int] = set()
    rotation_count = 0
    for index, instruction in enumerate(ir.instructions):
        _validate_bound_instruction(instruction)
        clauses = _validated_condition_clauses(instruction, index, measured)
        if instruction.name == "measure":
            if clauses:
                raise ValueError(
                    "conditional measurement is outside the Phase 7 profile"
                )
            if set(instruction.metadata) != {"is_dynamic", "classical_bit"} or (
                instruction.metadata.get("is_dynamic") is not True
            ):
                raise ValueError("measurement metadata is outside the Phase 7 profile")
            bit = instruction.metadata.get("classical_bit")
            if type(bit) is not int or bit < 0 or bit in measured:
                raise ValueError("measurement classical bits must be unique integers")
            measured.add(bit)
        elif instruction.name == "reset":
            if (
                clauses
                or set(instruction.metadata) != {"is_dynamic"}
                or (instruction.metadata.get("is_dynamic") is not True)
            ):
                raise ValueError("reset metadata is outside the fixed-round profile")
        elif instruction.name not in _ALLOWED_FIXED_GATES | _ALLOWED_ROTATIONS:
            raise ValueError(
                f"instruction {instruction.name!r} is outside the dynamic profile"
            )
        elif instruction.name in _ALLOWED_FIXED_GATES and (
            instruction.params or instruction.matrix is not None
        ):
            raise ValueError("Phase 7 fixed gates cannot carry parameters or matrices")
        elif instruction.name in _ALLOWED_ROTATIONS and (
            set(instruction.params) != {"theta"}
            or instruction.matrix is not None
            or not _is_real_scalar(instruction.params["theta"])
        ):
            raise ValueError(
                "Phase 8 rotations require one bound real scalar theta parameter"
            )
        elif set(instruction.metadata) - {"conditions", "condition_clauses"}:
            raise ValueError("gate metadata is outside the dynamic profile")
        if instruction.name in _ALLOWED_ROTATIONS:
            rotation_count += 1
    return_bit = ir.metadata.get("hybrid_dynamic_return_bit")
    if type(return_bit) is not int or return_bit not in measured:
        raise ValueError("hybrid dynamic return bit must name a measured classical bit")
    if ir.metadata.get("hybrid_dynamic_measurement_count") != len(measured):
        raise ValueError("hybrid dynamic measurement count metadata is inconsistent")
    if measured != set(range(len(measured))):
        raise ValueError("hybrid dynamic classical bits must be densely ordered")
    unrolled = ir.metadata.get("hybrid_dynamic_unrolled_iterations")
    if type(unrolled) is not int or unrolled < 0:
        raise ValueError("hybrid dynamic unrolled iteration count is invalid")
    parameter_order = ir.metadata.get("hybrid_parameter_order")
    if (
        not isinstance(parameter_order, (tuple, list))
        or any(not isinstance(name, str) for name in parameter_order)
        or len(set(parameter_order)) != len(parameter_order)
    ):
        raise ValueError("hybrid dynamic parameter order metadata is invalid")
    if len(parameter_order) != rotation_count:
        raise ValueError("hybrid dynamic parameter order does not match rotations")
    if ir.metadata.get("hybrid_stochastic_gradient_policy") != (
        "unsupported_fail_closed"
    ):
        raise ValueError("hybrid dynamic stochastic-gradient policy is unsupported")
    return ir, return_bit


def _is_real_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and value.dtype in {torch.float32, torch.float64}
    )


def execute_hybrid_dynamic_session(
    circuit_or_ir: Any,
    *,
    shots: int,
    seed: int | None = None,
    strategy: str = "auto",
    max_batched_bytes: int = 256 * 1024**2,
    noise_model: NoiseModel | None = None,
    _feedback_plan: DynamicFeedbackPlan | None = None,
) -> DynamicExecutionResult:
    """Execute one private, compiler-lowered local dynamic session."""

    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if seed is not None and type(seed) is not int:
        raise TypeError("seed must be an integer or None")
    ir, return_bit = _validate_session_ir(circuit_or_ir)
    circuit = DynamicCircuit(
        ir.n_wires,
        device="cpu",
        dtype=getattr(torch, ir.dtype),
    )
    circuit._instructions.extend(ir.instructions)
    result = run_dynamic(
        circuit,
        shots=shots,
        seed=seed,
        strategy=strategy,
        max_batched_bytes=max_batched_bytes,
        noise_model=noise_model,
        _feedback_plan=_feedback_plan,
    )
    statistics = dict(result.statistics)
    statistics.update(
        {
            "hybrid_dynamic_session": True,
            "returned_classical_bit": return_bit,
            "stochastic_gradient_policy": "unsupported_fail_closed",
        }
    )
    return replace(result, statistics=statistics)


__all__ = ("execute_hybrid_dynamic_session",)
