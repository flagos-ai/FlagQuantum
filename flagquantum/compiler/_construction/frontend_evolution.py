"""Capture static native Pauli sums using the existing ordered rotation effects."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...observables import Observable, _PauliTerm
from .. import _pauli_products, _pauli_schedule
from .._hybrid.model import Value
from .frontend_classical import classical_number
from .frontend_ports import _resolve
from .frontend_python import _error
from .frontend_views import gate_target, locations, static_integer
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
    destination, terms, time = _pauli_inputs(owner, operator, duration, target)
    order = static_integer(owner, order_node)
    repetitions = static_integer(owner, repetitions_node)
    if order not in {1, 2} or not 1 <= repetitions <= 256:
        _error(
            owner.source,
            node,
            "program.pauli",
            "use order 1 or 2 and 1..256 repetitions",
        )
    pending = _Evolution(node, destination, terms, time, order, repetitions)
    # Wider products keep the established streamed rotation path.
    if register_width(destination) > 3:
        flush_evolutions(owner)
        _emit_rotations(owner, pending)
        return
    if owner.evolutions and (
        owner.evolutions[-1].destination != destination or len(owner.evolutions) == 256
    ):
        flush_evolutions(owner)
    owner.evolutions.append(pending)


def capture_qdrift(owner: _Body, node: ast.Call) -> None:
    """Specialize deterministic sampling into the existing checked Pauli effects."""
    if len(node.args) not in {4, 5}:
        _error(
            owner.source,
            node,
            "program.signature",
            "qdrift takes operator, time, samples, target and optional seed",
        )
    operator, duration, sample_node, target = node.args[:4]
    samples = static_integer(owner, sample_node)
    seed = (
        classical_number(owner, node.args[4])
        if len(node.args) == 5
        else _pauli_schedule.DEFAULT_QDRIFT_SEED
    )
    if not 1 <= samples <= _pauli_schedule.MAX_QDRIFT_SAMPLES:
        _error(owner.source, sample_node, "program.pauli", "use 1..2048 samples")
    if type(seed) is not int or not isinstance(seed, int) or not 0 <= seed < 2**32:
        _error(
            owner.source,
            node,
            "program.pauli",
            "seed must be a static integer in 0..2**32-1",
        )
    destination, terms, time = _pauli_inputs(owner, operator, duration, target)
    if len(owner.operations) + samples > 4096:
        _error(
            owner.source,
            node,
            "budget",
            "sampled formula exceeds 4096 effect operations",
        )
    if not time or not any(coefficient for _, coefficient in terms):
        owner.emit("qreg.gate", node, (destination,), gate="rz", index=0, angle=0.0)
        return
    try:
        for index, angle in _pauli_schedule.sampled_pauli_steps(
            [coefficient for _, coefficient in terms], time, samples, seed
        ):
            owner.emit(
                "qreg.pauli_rotation",
                node,
                (destination,),
                support=terms[index][0],
                angle=angle,
            )
    except ValueError as error:
        _error(owner.source, node, "program.pauli", str(error))


def _pauli_inputs(
    owner: _Body, operator: ast.expr, duration: ast.expr, target: ast.expr
) -> tuple[Value, list[tuple[Support, float]], float]:
    targets = (
        tuple(gate_target(owner, item) for item in target.elts)
        if isinstance(target, (ast.List, ast.Tuple))
        else locations(owner, target)
    )
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(value != targets[0][0] for value, _ in targets)
    ):
        _error(
            owner.source,
            target,
            "program.view",
            "Pauli evolution requires distinct static locations in one register",
        )
    destination = targets[0][0]
    terms = [
        (tuple((targets[bit][1], axis) for bit, axis in support), coefficient)
        for support, coefficient in _observable_terms(owner, operator, len(targets))
    ]
    time = owner.phase_number(duration)
    if time is None:
        _error(
            owner.source,
            duration,
            "program.pauli",
            "evolution time must be a finite classical real",
        )
    return destination, terms, time


@dataclass(frozen=True)
class _Evolution:
    node: ast.Call
    destination: Value
    terms: list[tuple[Support, float]]
    time: float
    order: int
    repetitions: int

    @property
    def effects(self) -> int:
        return len(self.terms) * self.order * self.repetitions


def _emit_rotations(owner: _Body, evolution: _Evolution) -> None:
    if len(owner.operations) + evolution.effects > 4096:
        _error(
            owner.source,
            evolution.node,
            "budget",
            "product formula exceeds 4096 effect operations",
        )
    for index, step in _pauli_schedule.ordered_pauli_steps(
        len(evolution.terms), evolution.time, evolution.order, evolution.repetitions
    ):
        support, coefficient = evolution.terms[index]
        angle = step * coefficient
        if not math.isfinite(2 * angle):
            _error(
                owner.source,
                evolution.node,
                "program.pauli",
                "Pauli angle and doubled angle must be finite",
            )
        owner.emit(
            "qreg.pauli_rotation",
            evolution.node,
            (evolution.destination,),
            support=support,
            angle=float(angle),
        )


def flush_evolutions(owner: _Body) -> None:
    """Flush at effect/ownership barriers; never reorder intervening operations.

    Buffer at most 256 calls owned by one one-to-three-qubit register. Preserve
    small-program rotations. Only an otherwise over-budget sequence uses dense
    products, in the same 64-call chunks as the checked Qmod adapter.
    """
    pending, owner.evolutions = owner.evolutions, []
    if not pending:
        return
    if (
        len(pending) < 2
        or len(owner.operations) + sum(x.effects for x in pending) <= 4096
    ):
        for evolution in pending:
            _emit_rotations(owner, evolution)
        return
    destination = pending[0].destination
    width = register_width(destination)
    positions = {bit: bit for bit in range(width)}
    for start in range(0, len(pending), 64):
        matrix = _pauli_products.identity(1 << width)
        try:
            for evolution in pending[start : start + 64]:
                step = _pauli_products.pauli_step(
                    positions,
                    evolution.terms,
                    evolution.time / evolution.repetitions / evolution.order,
                    evolution.order,
                )
                matrix = _pauli_products.product(
                    _pauli_products.matrix_power(step, evolution.repetitions), matrix
                )
            matrix = _pauli_products.roundoff_unitary(matrix)
        except ValueError as error:
            _error(owner.source, pending[start].node, "program.pauli", str(error))
        owner.emit(
            "qreg.unitary",
            pending[start].node,
            (destination,) * width,
            indices=tuple(reversed(range(width))),
            matrix=tuple(tuple((v.real, v.imag) for v in row) for row in matrix),
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
        or not 1 <= len(raw.terms) <= 1024
    ):
        _error(
            owner.source,
            operator,
            "program.pauli",
            "use an immutable Observable with 1..1024 terms",
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
