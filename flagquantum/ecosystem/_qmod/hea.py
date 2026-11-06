"""Lower a bounded Qmod hardware-efficient ansatz to register gate effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .quantum_callbacks import Lambda
from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt
    from .phase import Location

_ONE = frozenset({"H", "X", "Z", "S", "I", "SX"})
_ONE_ANGLE = frozenset({"RX", "RY", "RZ", "PHASE"})
_TWO = frozenset({"CX", "CZ", "SWAP"})
_TWO_ANGLE = frozenset({"CRX"})


@dataclass(frozen=True)
class _Operand:
    gate: str
    axes: tuple[int, ...]
    parameterized: bool


def _operand(node: Expression, arity: int, flag: int) -> _Operand:
    if (
        not isinstance(node, Lambda)
        or len(node.parameters) != arity + 1
        or len({port.text for port in node.parameters}) != arity + 1
        or len(node.body) != 1
    ):
        reject(
            node.token, "hea.callback", "HEA operand requires one closed gate lambda"
        )
    (action,) = node.body
    gate = action.token.text
    angle, *qubits = (port.text for port in node.parameters)
    if (
        action.declaration is not None
        or gate
        not in (
            (_ONE_ANGLE if flag else _ONE)
            if arity == 1
            else (_TWO_ANGLE if flag else _TWO)
        )
        or len(action.arguments) != arity + flag
        or any(argument.children for argument in action.arguments)
    ):
        reject(
            action.token, "hea.callback", "gate must match operand arity and angle flag"
        )
    if flag and action.arguments[0].token.text != angle:
        reject(
            action.token, "hea.callback", "rotation must use its declared angle formal"
        )
    operands = tuple(argument.token.text for argument in action.arguments[flag:])
    if set(operands) != set(qubits):
        reject(
            action.token,
            "hea.callback",
            "operands must use distinct declared qubit formals",
        )
    return _Operand(
        gate.lower(), tuple(qubits.index(name) for name in operands), bool(flag)
    )


def lower_hea(owner: _Adapt, token: Token, arguments: tuple[Expression, ...]) -> None:
    if len(arguments) != 8:
        reject(token, "signature", "full_hea requires eight arguments")
    size, flags_arg, angles_arg, edges_arg, repeats_arg, one_arg, two_arg, target = (
        arguments
    )
    view = owner.array_view(target)
    width = owner.static_integer(size)
    repeats = owner.static_integer(repeats_arg)
    if width != len(view) or len(set(view)) != width or not 1 <= width <= 64:
        reject(
            size.token, "hea.width", "HEA width must match one to 64 disjoint qubits"
        )
    if not 0 <= repeats <= 128:
        reject(repeats_arg.token, "hea.budget", "HEA repetitions must be in 0..128")
    if (
        one_arg.token.text != "[]"
        or two_arg.token.text != "[]"
        or edges_arg.token.text != "["
    ):
        reject(
            token, "hea.signature", "HEA requires literal connectivity and gate lists"
        )
    flags = owner.integer_array(flags_arg)
    if len(flags) != len(one_arg.children) + len(two_arg.children) or any(
        flag not in {0, 1} for flag in flags
    ):
        reject(flags_arg.token, "hea.signature", "HEA flags must match the gate lists")
    split = len(one_arg.children)
    one = tuple(
        _operand(node, 1, flag)
        for node, flag in zip(one_arg.children, flags[:split], strict=True)
    )
    two = tuple(
        _operand(node, 2, flag)
        for node, flag in zip(two_arg.children, flags[split:], strict=True)
    )
    edges: list[tuple[Location, Location]] = []
    for edge in edges_arg.children:
        if len(edge.children) != 2:
            reject(
                edge.token,
                "hea.connectivity",
                "HEA connectivity requires pairs of indices",
            )
        a, b = (owner.static_integer(item) for item in edge.children)
        if not (0 <= a < width and 0 <= b < width and a != b):
            reject(
                edge.token,
                "hea.connectivity",
                "HEA edge endpoints must be distinct and in bounds",
            )
        edges.append((view[a], view[b]))
    edge_wires = {location for pair in edges for location in pair}
    if (
        sum(flags[:split]) > 1
        or sum(flags[split:]) > 1
        or (len(two) > 1 and len(edge_wires) != 2 * len(edges))
    ):
        reject(
            token,
            "hea.order",
            "HEA requires unambiguous angle and overlapping edge order",
        )
    angles = owner.real_array(angles_arg)
    expected = repeats * (sum(flags[:split]) * width + sum(flags[split:]) * len(edges))
    if len(angles) != expected:
        reject(
            angles_arg.token, "hea.binding", f"HEA requires exactly {expected} angles"
        )
    remaining = iter(angles)

    def emit(operands: tuple[_Operand, ...], locations: tuple[Location, ...]) -> None:
        for operand in operands:
            selected = tuple(locations[index] for index in operand.axes)
            registers, indices = zip(*selected, strict=True)
            attributes: dict[str, object] = {"gate": operand.gate}
            if operand.parameterized:
                attributes["angle"] = next(remaining)
            if len(selected) == 1:
                owner.effect_op(
                    "qreg.gate", token, registers, index=indices[0], **attributes
                )
            else:
                owner.effect_op(
                    "qreg.pair_gate", token, registers, indices=indices, **attributes
                )

    for _ in range(repeats):
        for location in view:
            emit(one, (location,))
        for edge_locations in edges:
            emit(two, edge_locations)
