"""Capture static raw-word rotation tables without numeric helper conversion."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_ports import _resolve
from .frontend_python import _error
from .frontend_views import gate_target, locations
from .rotation_table import emit_rotation_table

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_rotation_table(owner: _Body, node: ast.Call) -> None:
    if len(node.args) != 4:
        _error(
            owner.source,
            node,
            "program.rotation",
            "select_rotation takes axis, table, selector and target",
        )
    axis, table, source, target = node.args
    if not isinstance(axis, ast.Constant) or axis.value not in ("X", "Y", "Z"):
        _error(owner.source, axis, "program.rotation", "axis must be literal X, Y or Z")
    selector, destination = locations(owner, source), gate_target(owner, target)
    if (
        not 1 <= len(selector) <= 8
        or len(set(selector)) != len(selector)
        or destination in selector
    ):
        _error(
            owner.source,
            node,
            "program.rotation",
            "use one to eight distinct selector bits disjoint from the target",
        )
    angles: tuple[float, ...]
    if isinstance(table, (ast.List, ast.Tuple)):
        values = tuple(owner.phase_number(item) for item in table.elts)
        if any(value is None for value in values):
            _error(
                owner.source,
                table,
                "program.rotation",
                "angles must be finite static reals",
            )
        angles = tuple(value for value in values if value is not None)
    elif isinstance(table, ast.Name) and table.id in owner.classical:
        value = owner.classical[table.id]
        if not isinstance(value, tuple):
            _error(
                owner.source,
                table,
                "program.rotation",
                "angles require a flat real array",
            )
        angles = tuple(float(item) for item in value)
    elif isinstance(table, (ast.Name, ast.Attribute)):
        base: ast.expr = table
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and base.id in (
            owner.aliases if owner.aliases is not None else owner.ports
        ):
            _error(
                owner.source,
                table,
                "program.rotation",
                "angles cannot read quantum data",
            )
        raw = _resolve(owner.definition, table)
        if (
            type(raw) is not tuple
            or not isinstance(raw, tuple)
            or any(type(item) not in (int, float) for item in raw)
        ):
            _error(
                owner.source,
                table,
                "program.rotation",
                "static angle data must be an immutable real tuple",
            )
        try:
            angles = tuple(float(item) for item in raw)
        except OverflowError:
            _error(owner.source, table, "program.rotation", "angles must be finite")
    else:
        _error(
            owner.source,
            table,
            "program.rotation",
            "use a literal, bound or immutable real angle table",
        )
    if len(angles) != 1 << len(selector) or not all(
        math.isfinite(angle) for angle in angles
    ):
        _error(
            owner.source,
            table,
            "program.rotation",
            "finite table length must equal 2**selector width",
        )

    def emit(
        opcode: str, values: tuple[Value, ...], attributes: dict[str, object]
    ) -> None:
        owner.emit(opcode, node, values, **attributes)

    emit_rotation_table(axis.value, selector, destination, angles, emit)
