"""Fuse a proved local fixed-point value into a terminal rotation table."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from operator import add, mul, sub
from typing import TYPE_CHECKING

from .angles import REAL_LITERAL
from .selector_rotations import emit_rotation_table, is_selector_helper
from .syntax import Expression, Function, Parameter, Statement, Token, reject

if TYPE_CHECKING:
    from .context import _Adapt


def _plain(statement: Statement) -> bool:
    return statement == Statement(statement.token, statement.arguments)


def _match(node: Expression, name: Token) -> bool:
    return not node.children and node.token.text == name.text


def computed_output(function: Function) -> tuple[Parameter, Expression] | None:
    outputs = [port for port in function.parameters if port.role == "output"]
    if (
        not function.permutation
        or len(function.statements) != 1
        or len(outputs) != 1
        or not outputs[0].fractional_bits
        or outputs[0].kind != "qnum"
        or any(
            port.kind != "qnum"
            or port.record is not None
            or port.width is None
            or port.role not in {"const", "output"}
            or port.role == "const"
            and port.fractional_bits
            for port in function.parameters
        )
    ):
        return None
    statement = function.statements[0]
    if (
        not _plain(statement)
        or statement.token.text != "="
        or not _match(statement.arguments[0], outputs[0].name)
    ):
        return None
    expression = statement.arguments[1]
    exact_expression(
        expression,
        {port.name.text for port in function.parameters if port.role == "const"},
    )
    return outputs[0], expression


def _literal_angle(node: Expression) -> bool:
    if node.token.text == "-" and len(node.children) == 1:
        node = node.children[0]
    return not node.children and bool(REAL_LITERAL.fullmatch(node.token.text))


def reinterpreted_selector(
    function: Function, definitions: Mapping[str, Function]
) -> tuple[Parameter, Parameter, Statement] | None:
    if (
        function.permutation
        or len(function.parameters) != 2
        or len(function.statements) != 4
    ):
        return None
    reads = [port for port in function.parameters if port.role == "const"]
    writes = [port for port in function.parameters if port.role == "inout"]
    if len(reads) != 1 or len(writes) != 1:
        return None
    read, write = reads[0], writes[0]
    declaration, forward, selection, restore = function.statements
    view = declaration.declaration
    if (
        read.kind != "qnum"
        or not read.fractional_bits
        or write.kind != "qbit"
        or view is None
        or view.width != read.width
        or view.name.text in {read.name.text, write.name.text}
        or view.kind not in {"qnum", "qphase"}
        or not _plain(forward)
        or forward.token.text != "->"
        or not _plain(restore)
        or restore.token.text != "->"
        or not _plain(selection)
        or len(selection.arguments) != 2
        or len(forward.arguments) != 2
        or len(restore.arguments) != 2
        or not all(
            _match(node, name)
            for node, name in zip(
                (*forward.arguments, *restore.arguments),
                (read.name, view.name, view.name, read.name),
                strict=True,
            )
        )
    ):
        return None
    helper = definitions.get(selection.token.text)
    if helper is None or not is_selector_helper(helper) or len(helper.parameters) != 2:
        return None
    table = helper.statements[0]
    angles = table.arguments[1]
    if angles.token.text != "[]" or not all(
        _literal_angle(angle) for angle in angles.children
    ):
        return None
    actuals = dict(
        zip((p.name.text for p in helper.parameters), selection.arguments, strict=True)
    )
    if (
        not _match(actuals[table.arguments[2].token.text], view.name)
        or not _match(actuals[table.arguments[3].token.text], write.name)
        or any(
            port.role == "const" and port.width != read.width
            for port in helper.parameters
        )
    ):
        return None
    return read, write, table


_BINARY: dict[str, Callable[[Fraction, Fraction], Fraction]] = {
    "+": add,
    "-": sub,
    "*": mul,
    "min": min,
    "max": max,
}


def exact_expression(
    node: Expression,
    ports: set[str],
    constants: Mapping[Expression, Fraction] | None = None,
) -> Callable[[dict[str, Fraction]], Fraction]:
    if constants is not None and node in constants:
        literal = constants[node]
        return lambda inputs: literal
    text = node.token.text
    if not node.children:
        if text in ports:
            return lambda inputs: inputs[text]
        if REAL_LITERAL.fullmatch(text):
            literal = Fraction(text)
            if (
                max(literal.numerator.bit_length(), literal.denominator.bit_length())
                > 256
            ):
                reject(
                    node.token, "precision", "fixed-point coefficient exceeds 256 bits"
                )
            return lambda inputs: literal
    if text == "-" and len(node.children) == 1:
        operand = exact_expression(node.children[0], ports, constants)
        return lambda inputs: -operand(inputs)
    if text in _BINARY and len(node.children) == 2:
        left, right = (
            exact_expression(child, ports, constants) for child in node.children
        )
        operation = _BINARY[text]

        def checked(inputs: dict[str, Fraction]) -> Fraction:
            value = operation(left(inputs), right(inputs))
            if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
                reject(
                    node.token, "precision", "fixed-point intermediate exceeds 256 bits"
                )
            return value

        return checked
    reject(node.token, "precision", "unsupported exact fixed-point expression")


@dataclass(frozen=True)
class _ComposedRotation:
    local: Parameter
    expression: Expression
    compute: Statement
    select: Statement
    inputs: tuple[tuple[Parameter, Expression], ...]
    destination: Expression
    table: Statement


def composed_rotation(
    statements: tuple[Statement, ...], definitions: Mapping[str, Function]
) -> _ComposedRotation | None:
    if len(statements) != 3:
        return None
    declaration, compute, select = statements
    local = declaration.declaration
    if (
        local is None
        or not local.fractional_bits
        or not _plain(compute)
        or not _plain(select)
    ):
        return None
    producer = definitions.get(compute.token.text)
    consumer = definitions.get(select.token.text)
    if producer is None or consumer is None:
        return None
    computed = computed_output(producer)
    selected = reinterpreted_selector(consumer, definitions)
    if computed is None or selected is None:
        return None
    output, expression = computed
    reader, writer, table = selected
    if len(producer.parameters) != len(compute.arguments) or len(
        consumer.parameters
    ) != len(select.arguments):
        return None
    ports = dict(zip(producer.parameters, compute.arguments, strict=True))
    selector = dict(zip(consumer.parameters, select.arguments, strict=True))
    if not _match(ports[output], local.name) or not _match(
        selector[reader], local.name
    ):
        return None
    if (
        local.kind != "qnum"
        or local.width != output.width
        or local.width != reader.width
        or local.fractional_bits != output.fractional_bits
        or local.fractional_bits != reader.fractional_bits
    ):
        reject(
            local.name, "precision", "rotation local must have matching width and scale"
        )
    inputs = tuple((port, actual) for port, actual in ports.items() if port != output)
    return _ComposedRotation(
        local, expression, compute, select, inputs, selector[writer], table
    )


def lower_computed_rotation(owner: _Adapt, statements: tuple[Statement, ...]) -> bool:
    matched = composed_rotation(statements, owner.definitions)
    if matched is None:
        return False
    local = matched.local
    compute, select, table = matched.compute, matched.select, matched.table
    if owner.selector_guard is not None:
        reject(
            select.token,
            "rotation",
            "computed rotation cannot borrow an unrelated selector guard",
        )
    if local.name.text in owner.parameters or local.name.text in owner.constants:
        reject(local.name, "name", "rotation local must be fresh")

    operands = tuple(
        (port, owner.array_view(actual)) for port, actual in matched.inputs
    )
    if any(len(view) != port.width for port, view in operands):
        reject(compute.token, "width", "computed rotation input width disagrees")
    locations = tuple(location for _, view in operands for location in view)
    target = owner.gate_target(matched.destination)
    if not 1 <= len(locations) <= 8:
        reject(
            compute.token,
            "budget.rotation",
            "computed rotations need at most eight input bits",
        )
    if len(set(locations)) != len(locations) or target in locations:
        reject(
            select.token,
            "alias",
            "computed rotation inputs and target must be disjoint",
        )

    axis, entries = table.arguments[:2]
    angles = owner.real_array(entries)
    assert local.width is not None
    if axis.token.text not in {"X", "Y", "Z"} or len(angles) != 1 << local.width:
        reject(table.token, "rotation", "invalid rotation table for fixed-point word")
    evaluate = exact_expression(
        matched.expression, {port.name.text for port, _ in operands}
    )
    result = []
    for basis in range(1 << len(locations)):
        values: dict[str, Fraction] = {}
        offset = 0
        for port, view in operands:
            width = len(view)
            word = (basis >> offset) & ((1 << width) - 1)
            values[port.name.text] = Fraction(word, 1 << port.fractional_bits)
            offset += width
        index = evaluate(values) * (1 << local.fractional_bits)
        if index.denominator != 1 or not 0 <= index < len(angles):
            reject(
                compute.token,
                "precision",
                "fixed-point rotation index is not exactly representable",
            )
        result.append(angles[int(index)])
    original_locations = locations
    for bit in reversed(range(len(locations))):
        mask = 1 << bit
        if all(angle == result[index ^ mask] for index, angle in enumerate(result)):
            locations = (*locations[:bit], *locations[bit + 1 :])
            result = [angle for index, angle in enumerate(result) if not index & mask]
    retained = {value for value, _ in locations}
    for value, wire in original_locations:
        if value not in retained:
            # Resource pruning must not bypass the original initialization check.
            owner.effect_op(
                "qreg.control_phase",
                compute.token,
                (value,),
                indices=(wire,),
                angle=0.0,
            )
            retained.add(value)
    emit_rotation_table(
        owner, select.token, axis.token.text, locations, target, tuple(result)
    )
    return True
