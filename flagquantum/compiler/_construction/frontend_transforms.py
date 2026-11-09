"""Capture native Fourier and trigonometric transforms into checked register effects."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_python import _error
from .frontend_views import gate_target, locations
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
