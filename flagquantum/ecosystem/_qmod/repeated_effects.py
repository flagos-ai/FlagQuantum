"""Pack checked loop-independent reset cycles into a reusable register template."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compiler._construction.reset_cycles import reset_cycle_template
from .callback_arrays import capture_effects
from .syntax import Expression, Statement, reject
from .unitary_effects import PURE_EFFECTS

if TYPE_CHECKING:
    from .context import _Adapt


def lower_repeated_effects(owner: _Adapt, statement: Statement, count: int) -> bool:
    if count < 2 or owner.helper_stack or statement.repeat_index is None:
        return False
    parameter = statement.repeat_index.text
    callees: set[str] = set()
    found_reset = False
    reads_index = False

    def reads(node: Expression) -> bool:
        return (not node.children and node.token.text == parameter) or any(
            reads(child) for child in node.children
        )

    def visit(block: tuple[Statement, ...], at_call_site: bool) -> None:
        nonlocal found_reset, reads_index
        for action in block:
            if at_call_site:
                reads_index |= any(reads(node) for node in action.arguments)
            name = action.token.text
            found_reset |= name == "RESET"
            if name in owner.definitions and name not in callees:
                callees.add(name)
                visit(owner.definitions[name].statements, False)
            for scope in (
                action.within,
                action.apply,
                action.inverse_body,
                action.control_body,
                action.control_else,
                action.repeat_body,
                action.static_if_body,
            ):
                if scope:
                    visit(scope, at_call_site)

    visit(statement.repeat_body, True)
    if reads_index or not found_reset:
        return False
    effects = capture_effects(
        owner,
        statement.token,
        statement.repeat_body,
        allowed_effects=PURE_EFFECTS | {"qreg.reset"},
    )
    if not effects or not any(op.name == "qreg.reset" for op in effects):
        return False
    if len(effects) > 512:
        reject(
            statement.token, "budget", "a reusable reset cycle admits up to 512 effects"
        )
    token = statement.token
    name = (
        f"{owner.syntax.name.text}.$reset{token.location.line}.{token.location.column}"
    )
    template, owners = reset_cycle_template(name, effects)
    if any(function.name == name for function in owner.functions):
        reject(
            token,
            "name",
            "reusable reset scope must have one distinct lexical identity",
        )
    owner.functions.append(template)
    owner.effect_op("qreg.replay", token, owners, callee=name, count=count)
    return True
