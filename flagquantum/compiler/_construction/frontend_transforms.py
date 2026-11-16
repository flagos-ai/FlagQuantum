"""Capture native reversible register transforms into checked effects."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_python import _error
from .frontend_views import gate_target, locations, static_integer
from .program_types import qct_qst_type1, qct_qst_type2, qft, qst_type2
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_transform(owner: _Body, node: ast.Call, function: object) -> None:
    """Keep view identity checks in capture and transform legality in the verifier."""
    combined = function is qct_qst_type2
    expected = 2 if combined else 1
    if len(node.args) != expected:
        _error(
            owner.source,
            node,
            "program.signature",
            (
                "qft takes one register"
                if function is qft
                else f"transform takes {expected} argument(s)"
            ),
        )
    targets = locations(owner, node.args[0])
    destination = targets[0][0]
    if targets != tuple(
        (destination, bit) for bit in range(register_width(destination))
    ):
        _error(
            owner.source,
            node.args[0],
            "program.view",
            (
                "qft requires a complete register in logical bit order"
                if function is qft
                else "transform requires a complete register in logical bit order"
            ),
        )
    if function is qft:
        owner.emit("qreg.qft", node, (destination,))
        return
    if function is qct_qst_type1:
        owner.emit("qreg.type1_transform", node, (destination,))
        return
    kind = "combined" if combined else "sine" if function is qst_type2 else "cosine"
    operands: tuple[Value, ...] = (destination,)
    if combined:
        selector, index = gate_target(owner, node.args[1])
        if index != 0 or register_width(selector) != 1:
            _error(
                owner.source,
                node.args[1],
                "program.width",
                "combined transform selector must be one complete qubit",
            )
        operands += (selector,)
    owner.emit("qreg.type2_transform", node, operands, kind=kind)


def capture_modular_exp(owner: _Body, node: ast.Call) -> None:
    """Validate complete numeric operands before emitting the existing effect."""
    if len(node.args) != 4:
        _error(
            owner.source,
            node,
            "program.signature",
            "modular_exp takes modulus, base, target and exponent",
        )
    modulus, base = (static_integer(owner, arg) for arg in node.args[:2])
    modular_operands = []
    for argument in node.args[2:]:
        if (
            not isinstance(argument, ast.Name)
            or argument.id not in owner.ports
            or owner.ports[argument.id].kind not in {"qnum", "qbit"}
        ):
            _error(
                owner.source,
                argument,
                "program.modular_exp",
                "modular_exp requires complete numeric registers",
            )
        targets = locations(owner, argument)
        destination = targets[0][0]
        if targets != tuple(
            (destination, bit) for bit in range(register_width(destination))
        ):
            _error(
                owner.source,
                argument,
                "program.view",
                "modular_exp requires complete registers in logical bit order",
            )
        modular_operands.append(destination)
    owner.emit(
        "qreg.modular_exp",
        node,
        tuple(modular_operands),
        modulus=modulus,
        base=base,
    )
