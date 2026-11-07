"""Swap-test composition using existing indexed controlled-X effects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .syntax import Expression, Token, reject

if TYPE_CHECKING:
    from .context import _Adapt


def lower_swap_test(owner: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
    if len(args) != 3:
        reject(token, "signature", "swap_test requires two states and a fresh test bit")
    first, second = (owner.array_view(arg) for arg in args[:2])
    if not first or len(first) != len(second):
        reject(
            token, "swap_test.width", "swap_test inputs require equal nonzero widths"
        )
    if len(set((*first, *second))) != len(first) + len(second):
        reject(token, "swap_test.alias", "swap_test input locations must be disjoint")
    output = owner.atom(args[2])
    port = owner.parameters.get(output.text)
    if port is None or port.kind != "qbit" or port.role != "output":
        reject(
            output, "swap_test.output", "swap_test requires a fresh scalar qbit Output"
        )
    destination = owner.register(output, 1)
    control = (destination, 0)
    if control in (*first, *second):
        reject(
            token, "swap_test.alias", "swap_test output must be disjoint from inputs"
        )
    owner.effect_op("qreg.allocate", token, (destination,))
    owner.effect_op("qreg.gate", token, (destination,), gate="h", index=0)
    for left, right in zip(first, second, strict=True):
        owner.emit_indexed_x(token, (right, left))
        owner.emit_indexed_x(token, (control, left, right))
        owner.emit_indexed_x(token, (right, left))
    owner.effect_op("qreg.gate", token, (destination,), gate="h", index=0)
