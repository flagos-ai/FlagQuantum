"""Check and expand finite native unitary templates on a fixed logical layout."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ...core.ir import Instruction
from .._hybrid.model import Operation
from .profile import _fail

_MAX_OPERATIONS = 100_000
_ARITY = {
    "x": 1,
    "h": 1,
    "ry": 1,
    "rz": 1,
    "phase": 1,
    "cx": 2,
    "cphase": 2,
    "swap": 2,
    "ccx": 3,
}
_ANGLED = frozenset({"ry", "rz", "phase", "cphase"})
GateRow = tuple[str, tuple[int, ...], float | None]
GateTemplate = tuple[GateRow, ...]


def pack_gate_templates(
    instructions: Iterable[Instruction],
) -> tuple[tuple[GateTemplate, ...], tuple[int, ...]]:
    """Share exact native leaf runs separated by varying scalar Z rotations."""
    templates: dict[GateTemplate, int] = {}
    schedule: list[int] = []
    pending: list[GateRow] = []

    def add() -> None:
        if pending:
            template = tuple(pending)
            schedule.append(templates.setdefault(template, len(templates)))
            pending.clear()

    for gate in instructions:
        row = (gate.name, gate.wires, gate.params.get("theta"))
        if gate.name == "rz":
            add()
        pending.append(row)
        if gate.name == "rz" or len(pending) >= 1024:
            add()
    add()
    return tuple(templates), tuple(schedule)


def check_gate_tape(op: Operation, data_width: int, max_qubits: int) -> int:
    work = op.attributes["workspace"]
    templates = op.attributes["templates"]
    schedule = op.attributes["schedule"]
    if (
        type(work) is not int
        or not 0 <= work <= 12
        or not 1 <= data_width <= 16
        or data_width + work > max_qubits
        or not isinstance(templates, tuple)
        or not 1 <= len(templates) <= 512
        or not isinstance(schedule, tuple)
        or not 1 <= len(schedule) <= 512
        or any(
            type(index) is not int or not 0 <= index < len(templates)
            for index in schedule
        )
    ):
        _fail("register.gate_tape", "invalid bounded unitary template extent", op)
    limit = data_width + work
    for template in templates:
        if not isinstance(template, tuple) or not 1 <= len(template) <= 1024:
            _fail(
                "register.gate_tape",
                "unitary template must contain 1..1024 native gates",
                op,
            )
        for row in template:
            if not isinstance(row, tuple) or len(row) != 3:
                _fail("register.gate_tape", "invalid unitary template row", op)
            name, wires, angle = row
            if (
                not isinstance(name, str)
                or name not in _ARITY
                or not isinstance(wires, tuple)
                or len(wires) != _ARITY[name]
                or any(type(wire) is not int or not 0 <= wire < limit for wire in wires)
                or len(set(wires)) != len(wires)
                or (
                    name in _ANGLED
                    and (type(angle) is not float or not math.isfinite(angle))
                )
                or (name not in _ANGLED and angle is not None)
            ):
                _fail(
                    "register.gate_tape", "invalid native unitary gate in template", op
                )
    if sum(len(templates[index]) for index in schedule) > _MAX_OPERATIONS:
        _fail("register.budget", "unitary tape exceeds 100000 native gates", op)
    return work


def expand_gate_tape(
    op: Operation, physical: tuple[int, ...]
) -> tuple[Instruction, ...]:
    templates = tuple(
        tuple(
            Instruction(
                name,
                tuple(physical[wire] for wire in wires),
                {} if angle is None else {"theta": angle},
            )
            for name, wires, angle in template
        )
        for template in op.attributes["templates"]
    )
    return tuple(
        gate for index in op.attributes["schedule"] for gate in templates[index]
    )
