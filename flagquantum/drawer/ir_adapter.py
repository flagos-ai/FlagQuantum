"""Adapters from FlagQuantum circuit IR to drawer input objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.operator_schema import get_operator_schema


@dataclass(frozen=True)
class DrawableCircuit:
    """Minimal drawer-facing view shared by native Circuit, IR, and qdev inputs."""

    n_qubits: int
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
    qubits = list(getattr(instruction, "wires", ()) or ())
    params = getattr(instruction, "params", {}) or {}
    return {
        "name_or_mat": name,
        "name": name,
        "qubits": qubits,
        "params": _parameter_values(name, params),
        "matrix": getattr(instruction, "matrix", None),
        "metadata": dict(getattr(instruction, "metadata", {}) or {}),
    }


CANONICAL_QUBIT_KEY = "qubits"
"""The key every operation entry carries once it has passed this module."""

LEGACY_QUBIT_KEYS = ("wires",)
"""The key a pre-qubit-vocabulary object may use instead.

Read here and dropped on the way through, so no renderer ever meets it.
"""


def _op_qubits(op: Mapping[str, Any]) -> list[Any]:
    """The qubits an operation entry occupies, in either spelling.

    An entry produced by this package carries ``qubits``. A *legacy* qdev is not
    ours to rename -- it predates the qubit vocabulary and may expose ``wires``
    instead -- so both keys are read here, at the boundary, rather than in each
    of the ten places that consume an entry. The canonical key wins when an
    entry somehow carries both.
    """

    for key in (CANONICAL_QUBIT_KEY, *LEGACY_QUBIT_KEYS):
        if key in op:
            value = op[key]
            break
    else:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


def _normalize_op(op: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``op`` as a qubit-named entry, whatever the input called it.

    The legacy key is dropped rather than left alongside the canonical one: a
    renderer handed both spellings could read the stale one, and the reason to
    translate at the boundary is that nothing downstream has to know the other
    spelling ever existed.
    """

    normalized = {
        key: value for key, value in op.items() if key not in LEGACY_QUBIT_KEYS
    }
    normalized[CANONICAL_QUBIT_KEY] = _op_qubits(op)
    return normalized


def _qubit_count(program: Any) -> int | None:
    """The width ``program`` reports, under either spelling, or ``None``."""

    for name in ("n_qubits", "n_wires"):
        value = getattr(program, name, None)
        if isinstance(value, int):
            return value
    return None


def _detected_qubit_count(op_history: Sequence[Mapping[str, Any]]) -> int:
    """The width implied by the highest qubit any entry names."""

    highest = -1
    for op in op_history:
        for qubit in _op_qubits(op):
            if isinstance(qubit, int) and qubit > highest:
                highest = qubit
    return highest + 1 if highest >= 0 else 0


def to_drawable_circuit(program: Any) -> Any:
    """Return a drawer-compatible object for Circuit, CircuitIR, or operation data.

    Everything that leaves this function is qubit-named: the width is
    ``n_qubits`` and every operation entry carries ``qubits``. Anything a legacy
    or third-party object calls those two things is read here and nowhere else,
    so no renderer has to know that ``wires`` ever existed.
    """

    op_history = getattr(program, "op_history", None)
    if op_history is not None:
        entries = tuple(_normalize_op(op) for op in op_history)
        reported = _qubit_count(program)
        if reported is None:
            reported = _detected_qubit_count(entries)
        return DrawableCircuit(n_qubits=reported, op_history=entries)

    ir = program.to_ir() if hasattr(program, "to_ir") else program
    if not hasattr(ir, "instructions") or not hasattr(ir, "n_wires"):
        return program

    return DrawableCircuit(
        n_qubits=int(ir.n_wires),
        op_history=tuple(
            _instruction_to_op(instruction) for instruction in ir.instructions
        ),
    )


__all__ = [
    "CANONICAL_QUBIT_KEY",
    "LEGACY_QUBIT_KEYS",
    "DrawableCircuit",
    "to_drawable_circuit",
]
