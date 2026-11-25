"""Parse nested Qmod statement and scope grammar."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from .syntax_model import (
    _COMPARISONS,
    ANGLE_CALLS,
    MAX_PHASE_EXPRESSION_DEPTH,
    MAX_SCOPE_DEPTH,
    PREPARATIONS,
    Expression,
    Parameter,
    Statement,
    Token,
    reject,
)

if TYPE_CHECKING:
    from .syntax import Parser


def block(
    self: Parser,
    *,
    allow_scopes: bool = True,
    scope_depth: int = 0,
    power_body: bool = False,
    static_loop_body: bool = False,
    control_actions: bool = False,
    binding_preparation: bool = False,
    binding_actions: bool = False,
) -> tuple[Statement, ...]:
    if scope_depth > MAX_SCOPE_DEPTH:
        reject(
            self.current,
            "budget.depth",
            f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}",
        )
    self.take("{")
    statements: list[Statement] = []
    while self.current.text != "}":
        if self.current.text == "skip_control":
            token = self.take("skip_control")
            if scope_depth:
                reject(token, "scope", "skip_control table requires function scope")
            body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
            statements.append(Statement(token, (), skip_control=body))
            continue
        if self.current.text == "{" or self.tokens[self.position + 1].text == "->":
            if scope_depth and not binding_preparation:
                reject(self.current, "scope", "bindings require function scope")
            source = self.binding_group()
            operator = self.take("->")
            destination = self.binding_group()
            self.take(";")
            statements.append(Statement(operator, (source, destination)))
            continue
        if self.current.text == "invert" and self.tokens[self.position + 1].text == "{":
            token = self.take("invert")
            if scope_depth:
                if scope_depth >= MAX_SCOPE_DEPTH:
                    reject(
                        token,
                        "budget.depth",
                        f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}",
                    )
            body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
            statements.append(Statement(token, (), inverse_body=body))
            continue
        if self.current.text == "within" and self.tokens[self.position + 1].text == "{":
            from .bindings import parse_within

            statements.append(parse_within(self, allow_scopes, scope_depth))
            continue
        if (
            self.current.text in {"repeat", "power"}
            and self.tokens[self.position + 1].text == "("
        ):
            token = self.take(self.current.text)
            if scope_depth and not (
                static_loop_body or binding_actions or control_actions
            ):
                reject(
                    token,
                    "scope",
                    "static loops are admitted only at function scope",
                )
            self.take("(")
            if token.text == "repeat":
                index = self.name()
                self.take(":")
            else:
                index = Token(
                    f"$power{token.location.line}_{token.location.column}",
                    token.location,
                )
            count = self.expression(static=True)
            self.take(")")
            body = self.block(
                allow_scopes=allow_scopes,
                scope_depth=scope_depth + 1,
                static_loop_body=True,
                binding_actions=binding_actions,
            )
            statements.append(
                Statement(token, (count,), repeat_index=index, repeat_body=body)
            )
            continue
        if self.current.text == "foreach":
            token = self.take("foreach")
            self.take("(")
            names = [self.name()]
            while self.current.text == ",":
                self.take(",")
                names.append(self.name())
            self.take(":")
            matrix = self.expression(static=True, allow_slice=True)
            self.take(")")
            body = self.block(
                allow_scopes=False,
                scope_depth=scope_depth + 1,
                static_loop_body=True,
            )
            statements.append(
                Statement(token, (matrix,), repeat_body=body, matrix_row=tuple(names))
            )
            continue
        if self.current.text == "if" and self.tokens[self.position + 1].text == "(":
            token = self.take("if")
            self.take("(")
            if self.current.text in {"True", "False", "true", "false"}:
                literal = self.take(self.current.text)
                self.take(")")
                first = self.block(
                    allow_scopes=allow_scopes,
                    scope_depth=scope_depth + 1,
                    static_loop_body=static_loop_body,
                )
                second: tuple[Statement, ...] = ()
                if self.current.text == "else":
                    self.take("else")
                    second = self.block(
                        allow_scopes=allow_scopes,
                        scope_depth=scope_depth + 1,
                        static_loop_body=static_loop_body,
                    )
                selected = first if literal.text in {"True", "true"} else second
                if any(item.declaration is not None for item in (*first, *second)):
                    reject(
                        token,
                        "scope",
                        "static Boolean branches cannot declare local storage",
                    )
                statements.extend(selected)
                continue
            left = self.expression(static=True)
            comparison = self.current
            if (
                self.zero_walk_proof
                and comparison.text == ")"
                and left.token.text == "["
            ):
                self.take(")")
                body = self.block(
                    allow_scopes=allow_scopes,
                    scope_depth=scope_depth + 1,
                    static_loop_body=True,
                )
                statements.append(Statement(token, (left,), static_if_body=body))
                continue
            if comparison.text == ")" and not left.children:
                if scope_depth:
                    reject(
                        token,
                        "measurement",
                        "measurement feedback requires function scope",
                    )
                self.take(")")
                body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
                statements.append(
                    Statement(
                        replace(token, text="$measured_if"),
                        (left,),
                        control_body=body,
                    )
                )
                continue
            if comparison.text not in _COMPARISONS:
                reject(
                    comparison,
                    "expression",
                    "static if requires an integer comparison",
                )
            self.take(comparison.text)
            right = self.expression(static=True)
            self.take(")")
            body = self.block(
                allow_scopes=allow_scopes,
                scope_depth=scope_depth + 1,
                static_loop_body=True,
            )
            condition = Expression(comparison, (left, right))
            statements.append(Statement(token, (condition,), static_if_body=body))
            if self.current.text == "else":
                self.take("else")
                alternative = self.block(
                    allow_scopes=allow_scopes,
                    scope_depth=scope_depth + 1,
                    static_loop_body=True,
                )
                opposite = {
                    "==": "!=",
                    "!=": "==",
                    "<": ">=",
                    ">=": "<",
                    "<=": ">",
                    ">": "<=",
                }
                inverse = replace(comparison, text=opposite[comparison.text])
                statements.append(
                    Statement(
                        token,
                        (Expression(inverse, (left, right)),),
                        static_if_body=alternative,
                    )
                )
            continue
        if self.records and self.tokens[self.position + 1].text == ".":
            target_path = self.expression()
            op = self.current
            if op.text not in {"+=", "^="}:
                reject(op, "statement", "record field updates require += or ^=")
            self.take(op.text)
            right = self.expression(difference=True)
            self.take(";")
            statements.append(Statement(op, (target_path, right)))
            continue
        target = self.name()
        arguments: tuple[Expression, ...]
        if (
            target.text == "free"
            and not self.zero_walk_proof
            and scope_depth
            and not (allow_scopes and static_loop_body)
        ):
            reject(target, "scope", "free requires function scope")
        if (
            target.text in {"grover_operator", "grover_search", "phase_oracle"}
            and self.current.text == "("
        ):
            from .oracles import parse_oracle_call

            statements.append(parse_oracle_call(self, target, scope_depth))
            continue
        if (
            target.text == "apply_to_all"
            and self.tokens[self.position + 1].text == "lambda"
        ):
            self.take("(")
            self.take("lambda")
            self.take("(")
            parameter = self.name()
            self.take(")")
            body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
            self.take(",")
            target_array = self.call_argument()
            self.take(")")
            self.take(";")
            statements.append(
                Statement(target, (target_array,), map_callback=(parameter, body))
            )
            continue
        if target.text in {"qpe", "qpe_flexible"} and self.current.text == "(":
            from .qpe import parse_qpe_call

            statements.append(
                parse_qpe_call(
                    self,
                    target,
                    scope_depth,
                    reversible_preparation=binding_preparation,
                )
            )
            continue
        if self.current.text == ":":
            token = self.take(":")
            if scope_depth and not self.zero_walk_proof:
                reject(token, "scope", "declare locals at function level before within")
            record = None
            local_width: int | None
            local_length: int | None = None
            width_expression = None
            fractional_bits = 0
            if (
                self.current.text == "qnum"
                and self.tokens[self.position + 1].text == "<"
                and self.position + 3 < len(self.tokens)
                and self.tokens[self.position + 3].text == "."
                and not self.tokens[self.position + 2].text.isdecimal()
            ):
                self.take("qnum")
                self.take("<")
                width_expression = self.expression(static=True)
                if self.current.text == ",":
                    self.take(",")
                    sign = self.current
                    if sign.text not in {"UNSIGNED", "False"}:
                        reject(sign, "type", "derived width requires unsigned storage")
                    self.take(sign.text)
                    self.take(",")
                    self.take("0")
                self.take(">")
                kind, local_width = "qnum", None
            elif self.current.text == "bool":
                self.take("bool")
                kind, local_width = "bool", None
            elif self.current.text in self.records:
                record = self.name().text
                kind, local_width = "qbit[]", self.records[record].width
            else:
                kind, local_width, fractional_bits = self.scalar_type(
                    allow_fixed_array=True,
                    allow_qubit_array=True,
                    allow_phase=True,
                    allow_signed_integer=True,
                    allow_numeric_array=self.zero_walk_proof,
                )
                if self.zero_walk_proof and kind == "qnum" and self.current.text == "[":
                    self.take("[")
                    local_length = self.width()
                    self.take("]")
                    kind = "qnum[]"
                if kind == "qphase_signed":
                    reject(target, "type", "signed fixed-point locals are not admitted")
            self.take(";")
            statements.append(
                Statement(
                    token,
                    (),
                    declaration=Parameter(
                        target,
                        kind,
                        local_width,
                        "local",
                        record,
                        length=local_length,
                        width_expression=width_expression,
                        fractional_bits=fractional_bits,
                    ),
                )
            )
            continue
        if self.current.text == "[":
            bracket = self.take("[")
            callback_index = self.expression(static=True)
            self.take("]")
            if self.current.text == "^=":
                operator = self.take("^=")
                destination = Expression(bracket, (Expression(target), callback_index))
                value = self.expression(difference=True)
                self.take(";")
                statements.append(Statement(operator, (destination, value)))
            else:
                for part in ("(", ")", ";"):
                    self.take(part)
                statements.append(Statement(target, (), callback_index=callback_index))
            continue
        if self.current.text in {"=", "^=", "+="}:
            operator = self.take(self.current.text)
            if operator.text == "=" and self.current.text == "measure":
                if scope_depth:
                    reject(
                        operator,
                        "measurement",
                        "measure assignment requires function scope",
                    )
                operator = self.take("measure")
                self.take("(")
                measured = Expression(self.name())
                self.take(")")
                self.take(";")
                statements.append(
                    Statement(
                        replace(operator, text="$measure"),
                        (Expression(target), measured),
                    )
                )
                continue
            rhs = (
                self.signed_angle_argument()
                if operator.text == "^=" and self.current.text == "-"
                else self.expression(
                    decimal=self.closed_fixed_point or operator.text in {"=", "^="},
                    real=self.closed_fixed_point,
                    difference=True,
                    allow_slice=operator.text == "=",
                )
            )
            arguments = (Expression(target), rhs)
        elif self.current.text == "(":
            operator = target
            self.take("(")
            args: list[Expression] = []
            if self.current.text != ")":
                if target.text == "multi_suzuki_trotter":
                    args.append(self.literal_array(self.sparse_pauli_op))
                elif target.text in {"exponentiate", "qdrift", "suzuki_trotter"}:
                    if self.current.text == "SparsePauliOp":
                        args.append(self.sparse_pauli_op())
                    elif target.text == "suzuki_trotter" and self.current.text == "[":
                        args.append(self.pauli_terms())
                    else:
                        args.append(Expression(self.name()))
                elif target.text == "allocate" and self.current.text == "ceiling":
                    args.append(self.expression(static=True))
                elif target.text == "grover_diffuser" and self.current.text == "lambda":
                    from .quantum_callbacks import parse_lambda

                    args.append(parse_lambda(self, preparation=True))
                elif target.text == "prepare_exponential_state":
                    args.append(self.real_angle())
                elif target.text == "unitary":
                    args.append(
                        self.literal_array(
                            lambda: self.literal_array(self.matrix_entry)
                        )
                        if self.current.text == "["
                        else Expression(self.name())
                    )
                elif target.text == "linear_pauli_rotations":
                    args.append(self.literal_array(self.linear_rotation_basis))
                elif target.text == "select_rotation":
                    args.append(self.pauli_axis())
                elif target.text == "single_pauli_exponent":
                    args.append(self.literal_array(self.pauli_axis))
                elif target.text in PREPARATIONS:
                    args.append(
                        self.literal_array(
                            partial(
                                self.preparation_literal,
                                allow_fraction=target.text
                                in {"prepare_state", "inplace_prepare_state"},
                            )
                        )
                        if self.current.text == "["
                        else Expression(self.name())
                    )
                elif (
                    target.text == "control"
                    and self.zero_walk_proof
                    and self.current.text == "["
                ):
                    args.append(self.literal_array(self.call_argument))
                elif target.text == "control":
                    args.append(
                        self.expression(decimal=True, difference=True, allow_slice=True)
                    )
                elif target.text == "phase":
                    args.append(
                        self.expression(
                            decimal=True,
                            difference=True,
                            real=True,
                            depth_limit=MAX_PHASE_EXPRESSION_DEPTH,
                        )
                    )
                elif target.text in ANGLE_CALLS:
                    args.append(self.real_angle())
                else:
                    args.append(self.call_argument())
                while self.current.text == ",":
                    self.take(",")
                    if (
                        len(args) == 1
                        and target.text == "allocate"
                        and self.current.text in {"false", "true"}
                    ):
                        sign = self.take(self.current.text)
                        args.append(
                            Expression(replace(sign, text=sign.text.capitalize()))
                        )
                        continue
                    if (
                        len(args) == 2
                        and target.text == "allocate"
                        and self.current.text == "ceiling"
                    ):
                        args.append(self.expression(static=True))
                        continue
                    if (
                        len(args) == 3
                        and target.text == "suzuki_trotter"
                        and self.current.text == "ceiling"
                    ):
                        start = self.take("ceiling")
                        self.take("(")
                        scalar = self.real_angle()
                        self.take(")")
                        args.append(
                            Expression(
                                replace(start, text="$ceiling_repetitions"),
                                (scalar,),
                            )
                        )
                        continue
                    if len(args) == 3 and target.text == "full_hea":
                        args.append(
                            self.literal_array(
                                lambda: self.literal_array(
                                    lambda: self.expression(static=True)
                                )
                            )
                        )
                        continue
                    if self.zero_walk_proof and self.current.text == "lambda":
                        from .quantum_callbacks import Lambda

                        token = self.take("lambda")
                        self.take("(")
                        formal = self.name()
                        self.take(")")
                        args.append(
                            Lambda(token, parameters=(formal,), body=self.block())
                        )
                        continue
                    if self.zero_walk_proof and self.current.text == "[":
                        args.append(self.literal_array(lambda: Expression(self.name())))
                        continue
                    if len(args) == 1 and target.text == "multi_suzuki_trotter":
                        args.append(self.call_argument())
                        continue
                    if target.text == "linear_pauli_rotations" and len(args) in {
                        1,
                        2,
                    }:
                        args.append(self.literal_array(self.real_angle))
                        continue
                    if len(args) == 1 and target.text in {
                        "exponentiate",
                        "qdrift",
                        "prepare_amplitudes",
                        "prepare_state",
                        "suzuki_trotter",
                        "single_pauli_exponent",
                    }:
                        args.append(
                            self.signed_angle_argument()
                            if target.text in PREPARATIONS
                            else self.real_angle()
                        )
                        continue
                    args.append(
                        self.real_angle()
                        if (target.text == "U" and len(args) < 4)
                        or (len(args) == 1 and target.text in {"R", "phase"})
                        else self.call_argument()
                    )
            self.take(")")
            arguments = tuple(args)
        else:
            reject(
                self.current,
                "statement",
                "only fresh assignment, scalar declarations and admitted calls are supported",
            )
        if operator.text == "control" and self.current.text == "{":
            if scope_depth >= MAX_SCOPE_DEPTH:
                reject(
                    operator,
                    "budget.depth",
                    f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}",
                )
            body = self.block(
                allow_scopes=allow_scopes,
                scope_depth=scope_depth + 1,
                control_actions=True,
            )
            alternate = None
            if self.current.text == "else":
                self.take("else")
                alternate = self.block(
                    allow_scopes=allow_scopes,
                    scope_depth=scope_depth + 1,
                    control_actions=True,
                )
            statements.append(
                Statement(
                    operator, arguments, control_body=body, control_else=alternate
                )
            )
        else:
            self.take(";")
            statements.append(Statement(operator, arguments))
    self.take("}")
    return tuple(statements)
