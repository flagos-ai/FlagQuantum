"""Pack loop-invariant native reset effects into existing checked replay calls."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.unitary_effects import PURE_EFFECTS
from .._hybrid.verifier import HybridVerificationError
from .frontend_evolution import flush_evolutions
from .frontend_ports import _resolve
from .frontend_python import _error, _QuantumFunction
from .program_types import reset
from .reset_cycles import reset_cycle_template

if TYPE_CHECKING:
    from .frontend_program import _Body


def _contains_reset(
    definition: _QuantumFunction, nodes: list[ast.stmt], seen: set[int]
) -> bool:
    if len(seen) >= 32:
        return False
    found = False
    for statement in nodes:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                continue
            # Resolution is inspection of the retained namespace, never execution.
            try:
                function = _resolve(definition, node.func)
            except HybridVerificationError:
                continue
            if function is reset:
                found = True
            elif isinstance(function, _QuantumFunction) and id(function) not in seen:
                seen.add(id(function))
                found |= _contains_reset(function, function.source.node.body, seen)
    return found


def capture_reset_loop(owner: _Body, node: ast.For, sequence: range) -> bool:
    """Capture one invariant cycle without changing storage or execution budgets."""
    if len(sequence) < 2 or owner.region_depth or not isinstance(node.target, ast.Name):
        return False
    if any(
        isinstance(child, ast.Name) and child.id == node.target.id
        for statement in node.body
        for child in ast.walk(statement)
    ) or not _contains_reset(owner.definition, node.body, {id(owner.definition)}):
        return False
    flush_evolutions(owner)
    storage = (
        dict(owner.registers),
        dict(owner.ports),
        dict(owner.widths),
        set(owner.hidden),
        set(owner.dropped),
        dict(owner.classical),
    )
    start, effect = len(owner.operations), owner.effect
    functions = len(owner.owner.functions)
    steps, calls = owner.owner.loop_steps, owner.owner.inline_calls
    owner.loop_variables[node.target.id] = sequence[0]
    owner.region_depth += 1
    try:
        owner.statements(node.body, nested=True)
        flush_evolutions(owner)
    finally:
        owner.region_depth -= 1
        owner.loop_variables.pop(node.target.id, None)
    effects = tuple(owner.operations[start:])
    if not any(op.name == "qreg.reset" for op in effects):
        # A reset in an unselected static branch does not establish a reset cycle.
        from .frontend_views import expand_loop

        owner.owner.loop_steps += 1
        if owner.owner.loop_steps > 4096:
            _error(
                owner.source,
                node,
                "budget",
                "program exceeds 4096 expanded loop iterations",
            )
        expand_loop(owner, node, sequence[1:])
        return True
    if (
        len(effects) > 512
        or any(op.name not in PURE_EFFECTS | {"qreg.reset"} for op in effects)
        or functions != len(owner.owner.functions)
        or storage
        != (
            owner.registers,
            owner.ports,
            owner.widths,
            owner.hidden,
            owner.dropped,
            owner.classical,
        )
    ):
        _error(
            owner.source,
            node,
            "program.replay",
            "reset cycles require at most 512 effects on unchanged initialized storage",
        )
    owner.owner.loop_steps = steps + (owner.owner.loop_steps - steps + 1) * len(
        sequence
    )
    owner.owner.inline_calls = calls + (owner.owner.inline_calls - calls) * len(
        sequence
    )
    if (
        owner.owner.loop_steps > 4096
        or owner.owner.inline_calls > 256
        or owner.owner.serial >= 32
    ):
        _error(
            owner.source,
            node,
            "budget",
            "reset replay exceeds loop, helper or specialization budget",
        )
    name = f"{owner.name}.$reset{owner.owner.serial}"
    owner.owner.serial += 1
    template, arguments = reset_cycle_template(name, effects)
    owner.owner.functions.append(template)
    del owner.operations[start:]
    owner.effect = effect
    owner.emit("qreg.replay", node, arguments, callee=name, count=len(sequence))
    return True
