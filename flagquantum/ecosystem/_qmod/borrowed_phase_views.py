"""Verify temporary integer views over borrowed bits in diagonal Qmod helpers."""

from collections.abc import Mapping
from dataclasses import replace

from .syntax import Expression, Function, Statement, reject


def _diagonal_view(
    function: Function, constants: Mapping[str, Expression]
) -> tuple[Statement, ...] | None:
    if len(function.parameters) != 1 or len(function.statements) not in {2, 3}:
        return None
    (port,) = function.parameters
    declaration, scope, *suffix = function.statements
    alias = declaration.declaration
    if (
        not function.permutation
        or port.role != "const"
        or port.kind != "qbit[]"
        or port.record is not None
        or port.width is None
        or port.name.text == "pi"
        or alias is None
        or declaration != Statement(declaration.token, (), declaration=alias)
        or alias.role != "local"
        or alias.kind != "qnum"
        or alias.width != port.width
        or alias.name.text == port.name.text
        or alias.name.text in constants
        or alias.name.text == "pi"
        or (
            function.disable_const_checks is not None
            and function.disable_const_checks.text != port.name.text
        )
        or scope.within is None
        or len(scope.within) != 1
        or len(scope.apply) != 1
        or scope != Statement(scope.token, (), within=scope.within, apply=scope.apply)
    ):
        return None
    (binding,) = scope.within
    (control,) = scope.apply
    if (
        binding.token.text != "->"
        or binding != Statement(binding.token, binding.arguments)
        or len(binding.arguments) != 2
        or any(item.children for item in binding.arguments)
        or tuple(item.token.text for item in binding.arguments)
        != (port.name.text, alias.name.text)
        or control.token.text != "control"
        or len(control.arguments) != 1
        or control.control_body is None
        or control.control_else is not None
        or len(control.control_body) != 1
        or control
        != Statement(
            control.token, control.arguments, control_body=control.control_body
        )
    ):
        return None
    (condition,) = control.arguments
    if condition.token.text != "==" or len(condition.children) != 2:
        return None
    word, expected = condition.children
    if (
        word.children
        or word.token.text != alias.name.text
        or expected.children
        or not expected.token.text.isdecimal()
    ):
        return None
    for phase in (*control.control_body, *suffix):
        if (
            phase.token.text != "phase"
            or phase != Statement(phase.token, phase.arguments)
            or len(phase.arguments) != 1
        ):
            return None
        angle = phase.arguments[0]
        if angle.children or angle.token.text != "pi" or "pi" in constants:
            return None
    borrowed = Expression(
        replace(word.token, text="$predicate_word"), (Expression(port.name),)
    )
    tested = replace(condition, children=(borrowed, expected))
    return (replace(control, arguments=(tested,)), *suffix)


def expand_borrowed_phase_views(
    definitions: Mapping[str, Function], constants: Mapping[str, Expression]
) -> dict[str, Function]:
    rewritten = dict(definitions)
    for name, function in definitions.items():
        phase = _diagonal_view(function, constants)
        if phase is None:
            if function.disable_const_checks is not None:
                reject(
                    function.disable_const_checks,
                    "ownership",
                    "cannot prove annotated borrowed phase view",
                )
            continue
        rewritten[name] = replace(
            function,
            statements=phase,
            disable_const_checks=None,
            disable_permutation_check=None,
        )
    return rewritten
