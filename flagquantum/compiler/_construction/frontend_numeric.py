"""Native exact numeric assignment using the shared word compiler."""

from __future__ import annotations

import ast
import builtins
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, NoReturn

from .._hybrid.model import INDEX, HybridProgram, Operation, Value, ValueId
from .frontend_ports import _Port
from .frontend_python import _error
from .frontend_views import _require_builtin
from .numeric_expression import WordExpression
from .profile import uint_type
from .program_types import maximum, minimum
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def has_numeric(port: _Port) -> bool:
    return port.signed or bool(port.fractional_bits)


def needs_numeric(owner: _Body, expression: ast.expr) -> bool:
    return any(
        isinstance(node, (ast.Sub, ast.USub, ast.Call))
        or isinstance(node, ast.Constant)
        and type(node.value) is float
        or isinstance(node, ast.Name)
        and node.id in owner.ports
        and has_numeric(owner.ports[node.id])
        for node in ast.walk(expression)
    )


def has_numeric_reads(owner: _Body, expression: ast.expr | ast.stmt) -> bool:
    return any(
        isinstance(node, ast.Name)
        and node.id in owner.ports
        and has_numeric(owner.ports[node.id])
        for node in ast.walk(expression)
    )


def reject_numeric_reads(
    owner: _Body, expression: ast.expr | ast.stmt, context: str
) -> None:
    if has_numeric_reads(owner, expression):
        _error(
            owner.source,
            expression,
            "program.numeric",
            f"numeric reads are not admitted in {context}",
        )


_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.Mod: "%",
    ast.USub: "-",
    ast.Invert: "~",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _expression(owner: _Body, expression: ast.expr) -> WordExpression[ast.expr]:
    def reject(node: ast.expr, code: str, message: str) -> NoReturn:
        _error(owner.source, node, f"program.numeric.{code}", message)

    def text(node: ast.expr) -> str:
        if isinstance(node, ast.Name) and node.id in owner.ports:
            port = owner.ports[node.id]
            if port.kind == "qarray":
                reject(node, "format", "arithmetic reads require scalar inputs")
            owner.register(node)
            return node.id
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return str(node.value)
        if isinstance(node, ast.Call):
            if len(node.args) == 2 and not node.keywords:
                resolved = owner.resolve_callable(node.func)
                if resolved is maximum or resolved is minimum:
                    return "max" if resolved is maximum else "min"
                if isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}:
                    _require_builtin(owner, node.func, getattr(builtins, node.func.id))
                    return node.func.id
            reject(node, "call", "use a typed or unshadowed two-argument min/max")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = node.right
            if (
                not isinstance(exponent, ast.Constant)
                or type(exponent.value) is not int
                or not 1 <= exponent.value <= 8
            ):
                reject(
                    node, "power", "numeric powers require a literal exponent in 1..8"
                )
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            operator: ast.AST
            if isinstance(node, ast.Compare):
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    reject(
                        node, "operator", "chained numeric comparisons are not admitted"
                    )
                operator = node.ops[0]
            else:
                operator = node.op
            if type(operator) in _OPERATORS:
                return _OPERATORS[type(operator)]
        reject(node, "syntax", "unsupported exact numeric expression")

    def children(node: ast.expr) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.BinOp):
            return node.left, node.right
        if isinstance(node, ast.UnaryOp):
            return (node.operand,)
        if isinstance(node, ast.Compare):
            return (node.left, *node.comparators)
        if isinstance(node, ast.Call):
            return tuple(node.args)
        return ()

    name = f"{owner.name}.expr{owner.expressions}"
    owner.expressions += 1
    return WordExpression(
        expression,
        owner.widths,
        name,
        {name for name, port in owner.ports.items() if port.signed},
        text=text,
        children=children,
        location=owner.source.location,
        reject=reject,
        numeric_inputs={
            name: (port.signed, port.fractional_bits)
            for name, port in owner.ports.items()
        },
    )


def capture_numeric_expression(
    owner: _Body, node: ast.expr
) -> tuple[HybridProgram, tuple[Value, ...]]:
    """Bind exact owned scalar reads to the existing verified word graph."""
    expression = _expression(owner, node)
    graph = expression.function()
    owner.owner.functions.append(graph)
    return graph, tuple(owner.registers[name] for name in expression.names)


def assign_numeric(owner: _Body, node: ast.AugAssign) -> None:
    if not isinstance(node.target, ast.Name) or node.target.id not in owner.ports:
        _error(
            owner.source,
            node,
            "program.port",
            "numeric assignment requires a declared scalar port",
        )
    name = node.target.id
    port = owner.ports[name]
    if port.kind == "qarray" or not (port.role == "output" or name in owner.hidden):
        _error(
            owner.source,
            node,
            "program.numeric",
            "numeric assignment requires an owned scalar Output or hidden local",
        )
    if isinstance(node.op, ast.BitXor):
        destination = owner.register(node.target)
        mask = node.value.value if isinstance(node.value, ast.Constant) else None
        if (
            type(mask) is not int
            or not isinstance(mask, int)
            or not 0 <= mask < 1 << register_width(destination)
        ):
            _error(
                owner.source,
                node,
                "program.numeric.mask",
                "raw XOR requires a fitting nonnegative literal integer",
            )
        for bit in range(register_width(destination)):
            if mask >> bit & 1:
                owner.emit("qreg.gate", node, (destination,), gate="x", index=bit)
        # Even an all-zero mask must read initialized storage.
        if not mask:
            owner.emit("qreg.gate", node, (destination,), gate="i", index=0)
        return
    if not isinstance(node.op, ast.BitOr):
        _error(
            owner.source,
            node,
            "program.expression",
            "initialize fresh numeric outputs with |=",
        )
    explicit_format = name not in owner.inferred_numeric
    expression = _expression(owner, node.value)
    if not expression.quantum:
        value = Fraction(
            expression.integer_value(node.value),
            1 << expression.fractional_bits.get(node.value, 0),
        )
        fraction = (
            port.fractional_bits
            if explicit_format
            else expression.fractional_bits.get(node.value, 0)
        )
        scaled = value * (1 << fraction)
        if scaled.denominator != 1:
            _error(
                owner.source,
                node,
                "program.numeric.format",
                "initialization cannot discard fractional bits",
            )
        raw = scaled.numerator
        signed = port.signed if explicit_format else raw < 0
        width = owner.widths[name]
        if width is None:
            width = max(
                1,
                fraction,
                (~raw).bit_length() + 1 if raw < 0 else raw.bit_length() + int(signed),
            )
        low, high = (
            (-(1 << (width - 1)), (1 << (width - 1)) - 1)
            if signed
            else (0, (1 << width) - 1)
        )
        if not low <= raw <= high:
            _error(
                owner.source,
                node,
                "program.numeric.range",
                "initial value does not fit the declared numeric format",
            )
        owner.ports[name] = replace(port, signed=signed, fractional_bits=fraction)
        owner.emit(
            "qreg.initialize",
            node,
            (owner.register(node.target, width),),
            value=raw & ((1 << width) - 1),
        )
        return
    signed = node.value in expression.signed_words
    fraction = expression.fractional_bits.get(node.value, 0)
    if explicit_format and (port.signed, port.fractional_bits) != (
        signed,
        fraction,
    ):
        _error(
            owner.source,
            node,
            "program.numeric.format",
            "declared and inferred numeric formats must agree",
        )
    owner.ports[name] = replace(port, signed=signed, fractional_bits=fraction)
    graph = expression.function()
    block = graph.body.blocks[0]
    returned = block.operations[-1].operands[0]
    result_width = returned.type.parameters[0]
    width = max(result_width, fraction)
    destination = owner.register(node.target, width)
    if signed and register_width(destination) > result_width:
        extension_mask = Value(
            ValueId(
                graph.name,
                max(
                    v.id.index
                    for op in block.operations
                    for v in (*op.operands, *op.results)
                )
                + 1,
            ),
            INDEX,
        )
        extended = Value(
            ValueId(graph.name, extension_mask.id.index + 1),
            uint_type(register_width(destination)),
        )
        graph = replace(
            graph,
            body=replace(
                graph.body,
                blocks=(
                    replace(
                        block,
                        operations=(
                            *block.operations[:-1],
                            Operation(
                                "arith.constant",
                                results=(extension_mask,),
                                attributes={
                                    "value": (1 << register_width(destination)) - 1
                                },
                            ),
                            Operation(
                                "qint.bitwise",
                                (returned, extension_mask),
                                (extended,),
                                {"operator": "&", "signed": (True, False)},
                            ),
                            replace(block.operations[-1], operands=(extended,)),
                        ),
                    ),
                ),
            ),
        )
    owner.owner.functions.append(graph)
    actuals = tuple(owner.registers[name] for name in expression.names)
    owner.emit("qreg.compute", node, (destination, *actuals), callee=graph.name)
