"""Adapters from FlagQuantum circuit IR to drawer input objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.operator_schema import get_operator_schema


@dataclass(frozen=True)
class DrawableCircuit:
    """Minimal drawer-facing view shared by native Circuit, IR, and qdev inputs."""

    n_wires: int
    op_history: tuple[dict[str, Any], ...]


def _parameter_values(name: str, params: Mapping[str, Any] | None) -> list[Any]:
    if not params:
        return []
    schema = get_operator_schema(name)
    order = schema.parameters if schema is not None else ()
    if order and all(param_name in params for param_name in order):
        return [params[param_name] for param_name in order]
    return [params[key] for key in sorted(params)]


def _instruction_to_op(instruction: Any) -> dict[str, Any]:
    name = str(getattr(instruction, "name", ""))
    wires = list(getattr(instruction, "wires", ()) or ())
    params = getattr(instruction, "params", {}) or {}
    return {
        "name_or_mat": name,
        "name": name,
        "wires": wires,
        "params": _parameter_values(name, params),
        "matrix": getattr(instruction, "matrix", None),
        "metadata": dict(getattr(instruction, "metadata", {}) or {}),
    }


def to_drawable_circuit(program: Any) -> Any:
    """Return a drawer-compatible object for Circuit, CircuitIR, or operation data."""

    if hasattr(program, "op_history") and hasattr(program, "n_wires"):
        return program

    ir = program.to_ir() if hasattr(program, "to_ir") else program
    if not hasattr(ir, "instructions") or not hasattr(ir, "n_wires"):
        return program

    return DrawableCircuit(
        n_wires=int(ir.n_wires),
        op_history=tuple(
            _instruction_to_op(instruction) for instruction in ir.instructions
        ),
    )


__all__ = ["DrawableCircuit", "to_drawable_circuit"]
