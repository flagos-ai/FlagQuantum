"""Physical alias and workspace preconditions for a verified expression entry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from .._hybrid.model import HybridProgram, ValueId
from .profile import _fail, _positive_integer, _width, verify_construction


def verify_bindings(
    functions: Sequence[HybridProgram],
    inputs: Mapping[ValueId, Sequence[int]],
    output: Sequence[int],
    *,
    entry: str,
    n_wires: int,
    workspace: Sequence[int] = (),
    workspace_state: Literal[
        "fresh_zero", "caller_asserted_zero", "unknown"
    ] = "unknown",
) -> None:
    """Verify the call graph and disjoint bindings without inspecting quantum state.

    Fresh workspace is an allocator obligation; caller-asserted zero is a
    precondition, not a compiler proof. An XOR target need not initially be zero.
    """
    _positive_integer(n_wires, "n_wires")
    function = verify_construction(functions, entry=entry, max_width=n_wires)
    if workspace_state not in {"fresh_zero", "caller_asserted_zero", "unknown"}:
        _fail("workspace.state", "unknown workspace state declaration")
    block = function.body.blocks[0]
    if set(inputs) != {arg.id for arg in block.arguments}:
        _fail("binding.inputs", "input bindings must exactly cover entry arguments")
    groups = []
    for argument in block.arguments:
        wires = tuple(inputs[argument.id])
        if len(wires) != _width(argument.type, n_wires):
            _fail("binding.width", f"binding width disagrees with {argument.id}")
        groups.append(wires)
    result_type = block.operations[-1].operands[0].type
    if len(output) != _width(result_type, n_wires):
        _fail("binding.width", "output width disagrees with return type")
    groups.extend((tuple(output), tuple(workspace)))
    used: set[int] = set()
    for group in groups:
        for wire in group:
            if (
                isinstance(wire, bool)
                or not isinstance(wire, int)
                or not 0 <= wire < n_wires
            ):
                _fail("binding.wire", "wire must be an integer within the circuit")
            if wire in used:
                _fail(
                    "binding.alias",
                    f"physical wire {wire} occurs in overlapping bindings",
                )
            used.add(wire)
    if len(workspace) > 0 and workspace_state == "unknown":
        _fail(
            "workspace.precondition", "workspace requires fresh or caller-asserted zero"
        )
