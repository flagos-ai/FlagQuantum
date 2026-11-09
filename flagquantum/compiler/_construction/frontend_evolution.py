"""Capture static native Pauli sums using the existing ordered rotation effects."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from ...observables import Observable, _PauliTerm
from .. import _pauli_schedule
from .frontend_ports import _resolve
from .frontend_python import _error
from .frontend_views import locations, static_integer
from .pauli_rotation import Support
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_suzuki_trotter(owner: _Body, node: ast.Call) -> None:
    """Snapshot a bounded Observable; indices address logical target elements."""
    if len(node.args) != 5:
        _error(
            owner.source,
            node,
            "program.signature",
            "suzuki_trotter takes operator, time, order, repetitions and target",
        )
    operator, duration, order_node, repetitions_node, target = node.args
    targets = locations(owner, target)
    destination = targets[0][0]
    if targets != tuple(
        (destination, bit) for bit in range(register_width(destination))
    ):
        _error(
            owner.source,
            target,
            "program.view",
            "Pauli evolution requires a complete register in logical bit order",
        )
    terms = _observable_terms(owner, operator, len(targets))
    time = owner.phase_number(duration)
    if time is None:
        _error(
            owner.source,
            duration,
            "program.pauli",
            "evolution time must be a finite classical real",
        )
    order = static_integer(owner, order_node)
    repetitions = static_integer(owner, repetitions_node)
    if order not in {1, 2} or not 1 <= repetitions <= 256:
        _error(
            owner.source,
            node,
            "program.pauli",
            "use order 1 or 2 and 1..256 repetitions",
        )
    if len(owner.operations) + len(terms) * order * repetitions > 4096:
        _error(
            owner.source,
            node,
            "budget",
            "product formula exceeds 4096 effect operations",
        )
    for index, step in _pauli_schedule.ordered_pauli_steps(
        len(terms), time, order, repetitions
    ):
        term_support, coefficient = terms[index]
        angle = step * coefficient
        if not math.isfinite(2 * angle):
            _error(
                owner.source,
                duration,
                "program.pauli",
                "Pauli angle and doubled angle must be finite",
            )
        owner.emit(
            "qreg.pauli_rotation",
            node,
            (destination,),
            support=term_support,
            angle=float(angle),
        )


def _observable_terms(
    owner: _Body, operator: ast.expr, width: int
) -> list[tuple[Support, float]]:
    """Read canonical immutable Pauli data without invoking host protocols."""
    base = operator
    while isinstance(base, ast.Attribute):
        base = base.value
    visible = owner.aliases if owner.aliases is not None else owner.ports
    if (
        not isinstance(base, ast.Name)
        or base.id in visible
        or base.id in owner.classical
        or base.id in owner.loop_variables
    ):
        _error(
            owner.source, operator, "program.pauli", "use a static Observable binding"
        )
    raw = _resolve(owner.definition, operator)
    if (
        type(raw) is not Observable
        or not isinstance(raw, Observable)
        or type(raw.terms) is not tuple
        or not 1 <= len(raw.terms) <= 256
    ):
        _error(
            owner.source,
            operator,
            "program.pauli",
            "use an immutable Observable with 1..256 terms",
        )
    terms: list[tuple[Support, float]] = []
    for term in raw.terms:
        if (
            type(term) is not _PauliTerm
            or type(term.coefficient) not in (int, float)
            or abs(term.coefficient) > 2**53
            or not math.isfinite(term.coefficient)
            or type(term.factors) is not tuple
            or len(term.factors) > width
        ):
            _error(
                owner.source,
                operator,
                "program.pauli",
                "Pauli terms require finite built-in coefficients and immutable support",
            )
        support = []
        seen: set[int] = set()
        for pair in term.factors:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not int
                or not 0 <= pair[0] < width
                or pair[0] in seen
                or type(pair[1]) is not str
                or pair[1] not in {"x", "y", "z"}
            ):
                _error(
                    owner.source,
                    operator,
                    "program.pauli",
                    "Pauli indices must be distinct target elements with x/y/z axes",
                )
            seen.add(pair[0])
            support.append((pair[0], pair[1].upper()))
        terms.append((tuple(support), term.coefficient))
    return terms
