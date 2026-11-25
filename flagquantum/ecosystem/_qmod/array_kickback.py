"""Fold a checked borrowed comparison and nested X/H phase kickback."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from .phase_kickback import _atom, _plain, _tokens, _unary
from .syntax import _COMPARISONS, Expression, Function, Statement


def _nested_array_phase(
    wrapper: Function, definitions: Mapping[str, Function]
) -> tuple[Statement, set[str]] | None:
    if (
        not wrapper.permutation
        or len(wrapper.parameters) != 1
        or len(wrapper.statements) != 2
        or wrapper.disable_const_checks is not None
        or wrapper.disable_permutation_check is not None
    ):
        return None
    (read,) = wrapper.parameters
    declaration, scope = wrapper.statements
    marker = declaration.declaration
    if (
        read.role != "const"
        or read.kind != "qbit[]"
        or read.record is not None
        or read.width is None
        or marker is None
        or marker.kind != "qbit"
        or marker.role != "local"
        or declaration != Statement(declaration.token, (), declaration=marker)
        or read.name.text == marker.name.text
        or scope.within is None
        or len(scope.within) != 2
        or len(scope.apply) != 1
        or scope != Statement(scope.token, (), within=scope.within, apply=scope.apply)
    ):
        return None
    allocate, flip = scope.within
    if (
        not _plain(allocate)
        or allocate.token.text != "allocate"
        or len(allocate.arguments) not in {1, 2}
        or not _atom(allocate.arguments[-1], marker.name.text)
        or len(allocate.arguments) == 2
        and not _atom(allocate.arguments[0], "1")
        or not _unary(flip, "X", marker.name.text)
    ):
        return None
    (invoke,) = scope.apply
    inner = definitions.get(invoke.token.text)
    if (
        inner is None
        or not _plain(invoke)
        or len(inner.parameters) != 2
        or len(invoke.arguments) != 2
        or not inner.permutation
        or len({port.name.text for port in inner.parameters}) != 2
        or inner.disable_permutation_check is None
        or inner.disable_const_checks is None
        or len(inner.statements) != 1
    ):
        return None
    binding = {
        arg.token.text: port
        for port, arg in zip(inner.parameters, invoke.arguments, strict=True)
        if not arg.children
    }
    flag, borrowed = binding.get(marker.name.text), binding.get(read.name.text)
    if (
        flag is None
        or borrowed is None
        or flag.kind != "qbit"
        or flag.role != "const"
        or borrowed.kind != "qbit[]"
        or borrowed.role != "const"
        or borrowed.record is not None
        or borrowed.width != read.width
        or inner.disable_const_checks.text != flag.name.text
    ):
        return None
    (transform,) = inner.statements
    if (
        transform.within is None
        or len(transform.within) != 1
        or len(transform.apply) != 1
        or not _unary(transform.within[0], "H", flag.name.text)
        or transform
        != Statement(
            transform.token, (), within=transform.within, apply=transform.apply
        )
    ):
        return None
    (call,) = transform.apply
    consumed = {inner.name.text}
    flag_name, read_name = flag.name.text, borrowed.name.text
    while len(consumed) < 16 and _plain(call) and len(call.arguments) == 2:
        helper = definitions.get(call.token.text)
        if (
            helper is None
            or helper.name.text in consumed
            or not helper.permutation
            or helper.disable_const_checks is not None
            or helper.disable_permutation_check is not None
            or len(helper.parameters) != 2
            or len(helper.statements) != 1
            or len({port.name.text for port in helper.parameters}) != 2
        ):
            return None
        binding = {
            arg.token.text: port
            for port, arg in zip(helper.parameters, call.arguments, strict=True)
            if not arg.children
        }
        flag, borrowed = binding.get(flag_name), binding.get(read_name)
        if (
            flag is None
            or borrowed is None
            or flag.role != "inout"
            or flag.kind != "qbit"
            or borrowed.role != "const"
            or borrowed.kind not in {"qnum", "qbit[]"}
            or borrowed.record is not None
            or borrowed.width != read.width
            or borrowed.fractional_bits
        ):
            return None
        consumed.add(helper.name.text)
        flag_name, read_name = flag.name.text, borrowed.name.text
        (call,) = helper.statements
        if not _plain(call) or call.token.text != "^=" or len(call.arguments) != 2:
            continue
        target, condition = call.arguments
        if (
            not _atom(target, flag_name)
            or condition.token.text not in _COMPARISONS
            or len(condition.children) != 2
        ):
            return None
        word, value = condition.children
        if (
            not _atom(word, read_name)
            or value.children
            or not value.token.text.isdecimal()
        ):
            return None
        tested = Expression(
            replace(word.token, text="$predicate_word"), (Expression(read.name),)
        )
        predicate = replace(condition, children=(tested, value))
        phase = Statement(
            replace(invoke.token, text="phase"),
            (Expression(replace(invoke.token, text=repr(math.pi))),),
        )
        return (
            Statement(
                replace(invoke.token, text="control"),
                (predicate,),
                control_body=(phase,),
            ),
            consumed,
        )
    return None


def expand_nested_array_kickbacks(
    definitions: Mapping[str, Function],
) -> dict[str, Function]:
    """Fold a checked X/H preparation and borrowed comparison marker into phase."""
    rewritten = dict(definitions)
    removable: set[str] = set()
    for name, function in definitions.items():
        folded = _nested_array_phase(function, definitions)
        if folded is not None:
            phase, consumed = folded
            rewritten[name] = replace(function, statements=(phase,))
            removable.update(consumed)
    while removable:
        referenced = {
            token
            for function in rewritten.values()
            for token in _tokens(function.statements)
        }
        unused = removable - referenced
        if not unused:
            break
        for name in unused:
            rewritten.pop(name)
        removable.difference_update(unused)
    return rewritten
