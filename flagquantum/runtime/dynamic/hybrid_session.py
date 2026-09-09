"""Private Runtime handoff for compiler-lowered measurement-feedback sessions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from ...core.ir import CircuitIR, ensure_circuit_ir
from ._conditions import instruction_conditions
from .circuit import DynamicCircuit
from .execution import run_dynamic
from .result import DynamicExecutionResult

_ALLOWED_GATES = {"h", "x", "cx"}
_SESSION_METADATA = {
    "hybrid_dynamic_session",
    "hybrid_dynamic_return_bit",
    "hybrid_dynamic_measurement_count",
    "hybrid_stochastic_gradient_policy",
}


def _contains_trainable(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.requires_grad)
    if isinstance(value, Mapping):
        return any(_contains_trainable(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_trainable(item) for item in value)
    return False


def _validate_session_ir(circuit_or_ir: Any) -> tuple[CircuitIR, int]:
    ir = ensure_circuit_ir(circuit_or_ir)
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
    measured: set[int] = set()
    for index, instruction in enumerate(ir.instructions):
        if _contains_trainable(instruction.params) or _contains_trainable(
            instruction.matrix
        ):
            raise RuntimeError(
                "hybrid dynamic session stochastic gradients are unsupported"
            )
        conditions = instruction_conditions(instruction)
        if any(
            bit < 0 or expected not in {0, 1} for bit, expected in conditions
        ) or len({bit for bit, _ in conditions}) != len(conditions):
            raise ValueError(
                "classical conditions require unique bits and binary values"
            )
        if any(bit not in measured for bit, _ in conditions):
            raise ValueError(
                f"instruction {index} reads a classical bit before measurement"
            )
        if instruction.name == "measure":
            if conditions:
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
        elif instruction.name not in _ALLOWED_GATES:
            raise ValueError(
                f"instruction {instruction.name!r} is outside the Phase 7 profile"
            )
        elif instruction.params or instruction.matrix is not None:
            raise ValueError("Phase 7 fixed gates cannot carry parameters or matrices")
        elif set(instruction.metadata) - {"conditions"}:
            raise ValueError("fixed-gate metadata is outside the Phase 7 profile")
    return_bit = ir.metadata.get("hybrid_dynamic_return_bit")
    if type(return_bit) is not int or return_bit not in measured:
        raise ValueError("hybrid dynamic return bit must name a measured classical bit")
    if ir.metadata.get("hybrid_dynamic_measurement_count") != len(measured):
        raise ValueError("hybrid dynamic measurement count metadata is inconsistent")
    if measured != set(range(len(measured))):
        raise ValueError("hybrid dynamic classical bits must be densely ordered")
    if ir.metadata.get("hybrid_stochastic_gradient_policy") != (
        "unsupported_fail_closed"
    ):
        raise ValueError("hybrid dynamic stochastic-gradient policy is unsupported")
    return ir, return_bit


def execute_hybrid_dynamic_session(
    circuit_or_ir: Any,
    *,
    shots: int,
    seed: int | None = None,
    strategy: str = "auto",
    max_batched_bytes: int = 256 * 1024**2,
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
