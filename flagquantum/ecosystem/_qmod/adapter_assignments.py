"""Qmod assignments emitted through the function context."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.ranges import unsigned_width
from ...compiler._construction.register_profile import (
    register_width,
)
from .adapter_model import (
    _Expression,
    _literal_angle,
)
from .fixed_xor import lower_fixed_xor
from .parity_updates import lower_indexed_parity
from .syntax import (
    Expression,
    Statement,
    Token,
    reject,
)

if TYPE_CHECKING:
    from .context import _Adapt


def adapt_assignments(context: _Adapt, statement: Statement, index: str) -> None:
    token, args = statement.token, statement.arguments
    if token.text == "=":
        target = context.atom(args[0])
        if (
            context.helper_stack
            and not args[1].children
            and args[1].token.text in context.parameters
            and context.parameters[args[1].token.text].kind == "qnum"
            and args[1].token.text not in context.signed
            and context.parameters.get(target.text) is not None
            and context.parameters[target.text].role == "output"
            and context.parameters[target.text].kind == "qnum"
            and target.text not in context.signed
        ):
            view = context.array_view(args[1])
            register = context.register(target, len(view))
            if any(value == register for value, _ in view):
                reject(target, "alias", "copy requires disjoint input and Output")
            context.effect_op("qreg.allocate", token, (register,))
            for bit, control in enumerate(view):
                context.emit_indexed_x(token, (control, (register, bit)))
            return
        if "." in args[1].token.text and not args[1].children:
            reject(
                args[1].token,
                "operator",
                "decimal assignment requires min/max, subtraction or multiplication",
            )
        if (
            args[1].token.text == "-"
            and len(args[1].children) == 1
            and target.text in context.parameters
            and context.parameters[target.text].kind == "qbit"
        ):
            reject(args[1].token, "expression", "negation requires a numeric output")
        expression = _Expression(
            args[1], context.widths, f"{context.scope}.expr{index}", context.signed
        )
        if context.local_scope is not None and not expression.quantum:
            reject(
                token,
                "scope",
                "within preparation requires a quantum-dependent expression",
            )
        inferred_width = (
            expression.word_widths[args[1]]
            if args[1].token.text in {"%", "-"}
            else unsigned_width(expression.bound)
        )
        destination = context.register(target, inferred_width)
        if args[1] in expression.signed_words:
            if (
                context.scope != "main"
                or context.local_scope is not None
                or context.parameters[target.text].kind != "qnum"
                or register_width(destination) != expression.word_widths[args[1]]
            ):
                reject(
                    target,
                    "type",
                    "signed results require a matching-width main output",
                )
            context.signed.add(target.text)

        fractional_bits = expression.fractional_bits.get(args[1], 0)
        if fractional_bits:
            if (
                context.scope != "main"
                or context.local_scope is not None
                or context.parameters[target.text].kind != "qnum"
                or context.parameters[target.text].width is not None
            ):
                reject(
                    target,
                    "precision",
                    "fractional results require an inferred-width main qnum output",
                )
        if expression.quantum:
            graph = expression.function()
            context.functions.append(graph)
            inputs = tuple(
                context.register(context.parameters[name].name)
                for name in expression.names
            )
            context.effect_op(
                "qreg.compute", token, (destination, *inputs), callee=graph.name
            )
        else:
            context.effect_op(
                "qreg.initialize", token, (destination,), value=expression.bound
            )
        if fractional_bits:
            context.fractional_bits[target.text] = fractional_bits
    elif token.text == "$kickback_phase":
        condition, kickback_coefficient = args
        context.real_phase(
            token,
            Expression(replace(token, text="1")),
            kickback_coefficient,
            condition=condition,
            phase_sign=True,
        )
    elif token.text == "phase":
        default_coefficient = len(args) == 1
        if default_coefficient:
            args = (*args, Expression(Token("1.0", token.location)))
        if len(args) != 2 or "phase" in context.parameters:
            reject(
                token,
                "signature",
                "phase requires an expression and unshadowed pi coefficient",
            )
        pending = [args[0]]
        indexed = False
        while pending:
            node = pending.pop()
            if node.token.text == "[":
                indexed = True
                break
            pending.extend(node.children)
        if (
            default_coefficient
            or bool(args[1].children)
            or args[1].token.text.isdecimal()
            or indexed
            or args[1].token.text == "["
            or any(
                parameter.record is not None
                for parameter in context.parameters.values()
            )
            or args[1].token.text in context.classical_values
            or args[1].token.text in context.classical_integers
            or (
                context.helper_stack
                and (
                    context.classical_values
                    or context.classical_arrays
                    or context.classical_integers
                    or context.integer_arrays
                )
            )
        ):
            context.real_phase(token, args[0], args[1])
            return
        coefficient = args[1]
        angle = None
        if "." in coefficient.token.text:
            angle = _literal_angle(coefficient, token)
        elif (
            coefficient.children
            or coefficient.token.text != "pi"
            or "pi" in context.parameters
        ):
            reject(
                coefficient.token,
                "coefficient",
                "phase requires unshadowed pi or a decimal literal",
            )
        expression = _Expression(
            args[0], context.widths, f"{context.scope}.phase{index}", context.signed
        )
        if (
            args[0] in expression.signed_words
            or expression.fractional_bits.get(args[0], 0)
            or not expression.quantum
            or (angle is None and unsigned_width(expression.bound) != 1)
        ):
            reject(
                args[0].token,
                "phase",
                "phase requires a quantum expression and pi requires a one-bit result",
            )
        graph = expression.function()
        context.functions.append(graph)
        inputs = tuple(
            context.register(context.parameters[name].name) for name in expression.names
        )
        phase_attributes = {} if angle is None else {"angle": angle}
        context.effect_op(
            "qreg.phase", token, inputs, callee=graph.name, **phase_attributes
        )
    elif token.text == "^=":
        if not (
            lower_indexed_parity(context, token, args[0], args[1])
            or lower_fixed_xor(context, token, args[0], args[1])
        ):
            context.xor_update(token, args[0], args[1])
