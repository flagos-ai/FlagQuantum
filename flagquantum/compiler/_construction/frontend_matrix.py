"""Read bounded static matrix data without executing host expressions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .frontend_ports import _resolve
from .frontend_python import _error
from .frontend_views import gate_target, locations
from .register_profile import MAX_MATRIX_QUBITS

if TYPE_CHECKING:
    from .._hybrid.model import Value
    from .frontend_program import _Body


def capture_controlled_unitary(
    owner: _Body, condition: ast.expr, body: ast.Call
) -> None:
    """Retain a scalar control as the leading bit of a bounded static matrix."""
    inverted = isinstance(condition, ast.UnaryOp) and isinstance(
        condition.op, ast.Invert
    )
    selector = (
        condition.operand
        if inverted and isinstance(condition, ast.UnaryOp)
        else condition
    )
    capture_unitary(
        owner, body, control=gate_target(owner, selector), active=int(not inverted)
    )


def capture_unitary(
    owner: _Body,
    node: ast.Call,
    *,
    control: tuple[Value, int] | None = None,
    active: int = 1,
) -> None:
    """Reuse the checked unitary effect and its existing contract for indexed qubits."""
    if len(node.args) != 2:
        _error(
            owner.source, node, "program.signature", "unitary takes matrix and target"
        )
    matrix_node, target_node = node.args
    literal = isinstance(matrix_node, (ast.List, ast.Tuple))
    raw: object
    if literal:
        try:
            raw = ast.literal_eval(matrix_node)
        except (ValueError, TypeError, SyntaxError, RecursionError):
            _error(
                owner.source,
                matrix_node,
                "program.matrix",
                "use built-in numeric matrix literals",
            )
    elif isinstance(matrix_node, (ast.Name, ast.Attribute)):
        base: ast.expr = matrix_node
        while isinstance(base, ast.Attribute):
            base = base.value
        visible = owner.aliases if owner.aliases is not None else owner.ports
        if isinstance(base, ast.Name) and (
            base.id in visible
            or base.id in owner.classical
            or base.id in owner.loop_variables
        ):
            _error(
                owner.source,
                matrix_node,
                "program.matrix",
                "matrix data must be a static immutable tuple",
            )
        raw = _resolve(owner.definition, matrix_node)
    else:
        _error(
            owner.source,
            matrix_node,
            "program.matrix",
            "use a matrix literal or static immutable tuple",
        )
    containers = (list, tuple) if literal else (tuple,)
    limit = 1 << MAX_MATRIX_QUBITS
    if (
        type(raw) not in containers
        or not isinstance(raw, (list, tuple))
        or not 2 <= len(raw) <= limit
    ):
        _error(
            owner.source,
            matrix_node,
            "program.matrix",
            "matrix needs 2..32 rows in a literal or immutable tuple",
        )
    prepared = []
    for row in raw:
        if (
            type(row) not in containers
            or not isinstance(row, (list, tuple))
            or not 2 <= len(row) <= limit
        ):
            _error(
                owner.source,
                matrix_node,
                "program.matrix",
                "matrix rows need 2..32 built-in entries",
            )
        entries = []
        for entry in row:
            if type(entry) not in (int, float, complex):
                _error(
                    owner.source,
                    matrix_node,
                    "program.matrix",
                    "matrix entries must be built-in numbers, excluding bool",
                )
            try:
                number = complex(entry)
            except OverflowError:
                _error(
                    owner.source,
                    matrix_node,
                    "program.matrix",
                    "matrix entry exceeds finite numeric range",
                )
            entries.append((float(number.real), float(number.imag)))
        prepared.append(tuple(entries))
    if isinstance(target_node, (ast.List, ast.Tuple)):
        if not 1 <= len(target_node.elts) <= MAX_MATRIX_QUBITS:
            _error(
                owner.source,
                target_node,
                "program.matrix",
                "matrix targets require one to five qubits",
            )
        targets = tuple(
            reversed(tuple(gate_target(owner, item) for item in target_node.elts))
        )
    else:
        targets = tuple(reversed(locations(owner, target_node)))
    if not 1 <= len(targets) <= MAX_MATRIX_QUBITS:
        _error(
            owner.source,
            target_node,
            "program.matrix",
            "matrix targets require one to five qubits",
        )
    if control is not None:
        if control in targets:
            _error(
                owner.source,
                target_node,
                "program.alias",
                "matrix control and target must be disjoint",
            )
        if len(targets) + 1 > MAX_MATRIX_QUBITS:
            _error(
                owner.source,
                target_node,
                "program.matrix",
                "controlled matrix including selector requires at most five qubits",
            )
        size = 1 << len(targets)
        if len(prepared) != size or any(len(row) != size for row in prepared):
            _error(
                owner.source,
                matrix_node,
                "program.matrix",
                "matrix dimensions must match the target width",
            )
        prepared = [
            tuple(
                (
                    prepared[row % size][column % size]
                    if row // size == column // size == active
                    else (float(row == column and row // size != active), 0.0)
                )
                for column in range(2 * size)
            )
            for row in range(2 * size)
        ]
        targets = (control, *targets)
    owner.emit(
        "qreg.unitary",
        node,
        tuple(value for value, _ in targets),
        matrix=tuple(prepared),
        indices=tuple(index for _, index in targets),
    )
