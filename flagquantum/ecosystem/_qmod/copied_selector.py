"""Borrow a proved temporary slice copy only for a terminal read-only selector."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .syntax import Expression, Function, Parameter, Statement, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


def slice_output(function: Function) -> tuple[Parameter, Parameter, Expression] | None:
    sources = [p for p in function.parameters if p.role == "const"]
    outputs = [p for p in function.parameters if p.role == "output"]
    if (
        not function.permutation
        or len(function.statements) != 1
        or len(sources) != 1
        or len(outputs) != 1
        or len({port.name.text for port in function.parameters})
        != len(function.parameters)
        or any(
            port not in sources + outputs
            and (port.role != "integer" or port.kind != "int")
            for port in function.parameters
        )
    ):
        return None
    source, output = sources[0], outputs[0]
    statement = function.statements[0]
    if (
        source.kind != "qbit[]"
        or source.record is not None
        or output.kind != "qbit[]"
        or output.record is not None
        or statement.token.text != "="
        or len(statement.arguments) != 2
        or statement != Statement(statement.token, statement.arguments)
    ):
        return None
    name, view = statement.arguments
    if name.children or name.token.text != output.name.text or view.token.text != ":":
        return None
    actual = view.children[0]
    if actual.children or actual.token.text != source.name.text:
        return None
    return source, output, view


def lower_copied_selector(
    owner: _Adapt, statements: tuple[Statement, ...], index: str
) -> bool:
    declaration, call, loop = statements
    local = declaration.declaration
    helper = owner.definitions.get(call.token.text)
    if (
        local is None
        or local.kind != "qnum"
        or local.fractional_bits
        or helper is None
        or call != Statement(call.token, call.arguments)
        or loop.repeat_index is None
        or len(loop.repeat_body) != 1
        or len(call.arguments) != len(helper.parameters)
    ):
        return False
    matched = slice_output(helper)
    if matched is None:
        return False
    source, output, view = matched
    ports = dict(zip(helper.parameters, call.arguments, strict=True))
    if ports[output].children or ports[output].token.text != local.name.text:
        return False
    guarded = loop.repeat_body[0]
    if (
        guarded.control_body is None
        or len(guarded.control_body) != 1
        or guarded.control_else is not None
        or guarded.control_body[0].token.text != "^="
    ):
        return False
    destination = guarded.control_body[0].arguments[0]
    output_word = owner.parameters.get(destination.token.text)
    if destination.children or output_word is None or output_word.kind != "qphase":
        return False
    condition = guarded.arguments[0]
    if (
        condition.token.text != "=="
        or len(condition.children) != 2
        or condition.children[0].children
        or condition.children[0].token.text != local.name.text
    ):
        return False
    actual = owner.array_view(ports[source], allow_phase=True)
    if source.width is not None and len(actual) != source.width:
        reject(call.token, "width", "slice copy source width disagrees")
    values = {
        port.name.text: owner.static_integer(arg)
        for port, arg in ports.items()
        if port.role == "integer"
    }
    name = local.name.text
    if name in owner.parameters or name in owner.constants:
        reject(local.name, "name", "copied selector name must be fresh")

    def bind(node: Expression) -> Expression:
        if not node.children and node.token.text in values:
            value = values[node.token.text]
        elif node.token.text == "." and tuple(
            item.token.text for item in node.children
        ) == (source.name.text, "len"):
            value = len(actual)
        elif not node.children and not node.token.text.isdecimal():
            reject(node.token, "scope", "slice copy bounds may only use bound formals")
        else:
            return replace(node, children=tuple(bind(item) for item in node.children))
        literal = Expression(replace(node.token, text=str(abs(value))))
        return (
            literal
            if value >= 0
            else Expression(replace(node.token, text="-"), (literal,))
        )

    start, stop = (owner.static_integer(bind(arg)) for arg in view.children[1:])
    if not 0 <= start < stop <= len(actual):
        reject(view.token, "width", "copied selector slice is out of bounds")
    if any(
        width is not None and width != stop - start
        for width in (output.width, local.width)
    ):
        reject(view.token, "width", "copied selector slice width disagrees")
    # The table consumer reads only this alias and verifies it is disjoint
    # from the writable word, so the coherent slice copy and its cleanup cancel.
    owner.parameters[name] = local
    owner.widths[name] = stop - start
    owner.views[name] = actual[start:stop]
    try:
        owner.statement(loop, index)
    finally:
        owner.views.pop(name)
        owner.widths.pop(name)
        owner.parameters.pop(name)
    return True
