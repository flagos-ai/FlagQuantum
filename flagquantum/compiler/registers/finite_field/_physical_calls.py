"""Instruction ranges for the finite-field generator's nested physical calls."""

from dataclasses import replace

from ....core.ir import Instruction
from ..._component_trace.trace import ComponentCall


def append_field_call(
    gates: list[Instruction],
    calls: list[ComponentCall] | None,
    name: str,
    operations: tuple[Instruction, ...],
    children: tuple[ComponentCall, ...] = (),
) -> None:
    """Append gates and rebind relative call ranges at the insertion point."""
    start = len(gates)
    gates.extend(operations)
    if calls is None:
        return

    def rebind(call: ComponentCall) -> ComponentCall:
        return replace(
            call,
            start=call.start + start,
            stop=call.stop + start,
            children=tuple(rebind(child) for child in call.children),
        )

    calls.append(
        ComponentCall(name, start, len(gates), children=tuple(map(rebind, children)))
    )
