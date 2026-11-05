"""Checked adapter-local expansion of borrowed phase functions and Grover rounds."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from .integers import checked_integer
from .syntax import Expression, Function, Parameter, Record, Statement, Token, reject

if TYPE_CHECKING:
    from .syntax import Parser

MAX_ORACLE_EXPANSIONS = 4096


def static_index(
    node: Expression, integers: Mapping[str, Expression], width: int | None
) -> int | None:
    if not node.children:
        if node.token.text.isdecimal():
            return checked_integer(int(node.token.text), node.token)
        if node.token.text in {item.token.text for item in integers.values()}:
            return None
    elif node.token.text == "." and node.children[1].token.text == "len":
        return width
    elif node.token.text in {"+", "-", "*"} and len(node.children) == 2:
        left, right = (static_index(child, integers, width) for child in node.children)
        if left is None or right is None:
            return None
        value = (
            left + right
            if node.token.text == "+"
            else left - right if node.token.text == "-" else left * right
        )
        return checked_integer(value, node.token)
    reject(
        node.token,
        "oracle.alias",
        "phase oracle index requires bounded closed integer arithmetic",
    )


def parse_oracle_call(parser: Parser, target: Token, scope_depth: int) -> Statement:
    """Read the static callback form within the shared bounded source parser."""
    if target.text == "phase_oracle":
        from .predicates import parse_phase_oracle_call

        return parse_phase_oracle_call(parser, target, scope_depth)
    parser.take("(")
    repetitions = None
    if target.text == "grover_search":
        repetitions = parser.call_argument()
        parser.take(",")
    callback = None
    if parser.current.text == "lambda":
        marker = parser.take("lambda")
        parser.take("(")
        parameter = parser.name()
        parser.take(")")
        callback = (
            parameter,
            parser.block(allow_scopes=False, scope_depth=scope_depth + 1),
        )
        oracle = Expression(marker)
    else:
        oracle = Expression(parser.name())
    parser.take(",")
    preparation = Expression(Token("hadamard_transform", target.location))
    if repetitions is None:
        preparation = Expression(parser.name())
        parser.take(",")
    destination = parser.call_argument()
    parser.take(")")
    parser.take(";")
    return Statement(
        target,
        (
            (oracle, preparation, destination)
            if repetitions is None
            else (repetitions, oracle, destination)
        ),
        oracle_callback=callback,
    )


def expand_static_oracles(
    definitions: Mapping[str, Function],
    constants: Mapping[str, Expression],
    records: Mapping[str, Record],
) -> dict[str, Function]:
    """Erase checked static callbacks before constructing any owned Compiler IR.

    The profile is one whole borrowed array, with diagonal phase effects only.
    General callable values, quantum captures and consuming ports are refused.
    All bodies are checked before expansion, including unused and zero-power ones.
    """
    oracles = {
        name: function
        for name, function in definitions.items()
        if any(p.role == "const" and p.kind == "qbit[]" for p in function.parameters)
    }
    count = 0

    def borrowed(function: Function) -> Parameter:
        if (
            sum(p.role == "const" and p.kind == "qbit[]" for p in function.parameters)
            != 1
            or any(
                not (p.role == "const" and p.kind == "qbit[]")
                and not (p.role == "integer" and p.kind == "int")
                for p in function.parameters
            )
            or len({p.name.text for p in function.parameters})
            != len(function.parameters)
        ):
            reject(
                function.name,
                "oracle",
                "phase oracle requires one Const array and distinct integer ports; Output and alias ports cannot escape",
            )
        return next(p for p in function.parameters if p.role == "const")

    def closed(
        expression: Expression,
        variable: str,
        integers: Mapping[str, Expression],
        width: int | None,
        record: str | None,
    ) -> Expression:
        if not expression.children:
            name = expression.token.text
            if name in constants:
                return constants[name]
            if name in integers:
                return integers[name]
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) and name not in {
                variable,
                "pi",
            }:
                reject(
                    expression.token,
                    "oracle.capture",
                    "phase callback cannot capture another quantum value or escape its target",
                )
        if expression.token.text == ".":
            root = expression.children[0]
            member = (
                expression.children[1]
                if len(expression.children) == 2
                else Expression(Token("len", expression.token.location))
            )
            if (
                record is None
                and not root.children
                and root.token.text == variable
                and not member.children
                and member.token.text == "len"
            ):
                return (
                    replace(expression, children=(root, member))
                    if width is None
                    else Expression(Token(str(width), expression.token.location))
                )
            if (
                record is None
                or root.children
                or root.token.text != variable
                or member.children
                or member.token.text
                not in {field.name.text for field in records[record].fields}
            ):
                reject(
                    expression.token,
                    "oracle.capture",
                    "field access requires a declared field of the borrowed record",
                )
            return expression
        children = tuple(
            closed(child, variable, integers, width, record)
            for child in expression.children
        )
        if expression.token.text == "[":
            array, offset = children
            if array.children or array.token.text != variable:
                reject(
                    expression.token,
                    "oracle.alias",
                    "phase oracle indexing requires its whole borrowed target",
                )
            index = static_index(offset, integers, width)
            if index is not None and (
                index < 0 or (width is not None and index >= width)
            ):
                reject(
                    offset.token, "oracle.alias", "phase oracle index is out of bounds"
                )
            if index is not None and offset.children:
                children = (array, Expression(Token(str(index), offset.token.location)))
        return replace(
            expression,
            children=children,
        )

    def substitute(
        expression: Expression, variable: str, target: Expression
    ) -> Expression:
        if not expression.children and expression.token.text == variable:
            return target
        if expression.token.text == ".":
            return replace(
                expression,
                children=(
                    substitute(expression.children[0], variable, target),
                    expression.children[1],
                ),
            )
        return replace(
            expression,
            children=tuple(
                substitute(child, variable, target) for child in expression.children
            ),
        )

    def body(
        statements: tuple[Statement, ...],
        variable: str,
        target: Expression,
        active: tuple[str, ...],
        integers: Mapping[str, Expression],
        width: int | None,
        record: str | None,
    ) -> tuple[Statement, ...]:
        nonlocal count
        result: list[Statement] = []
        for statement in statements:
            count += 1
            if count > MAX_ORACLE_EXPANSIONS:
                reject(
                    statement.token,
                    "budget.oracle",
                    "phase oracle expansion exceeds 4096 statements",
                )
            token = statement.token
            if (
                statement.declaration is not None
                or statement.within is not None
                or statement.inverse_body is not None
                or statement.repeat_index is not None
                or statement.static_if_body is not None
                or statement.map_callback is not None
                or statement.oracle_callback is not None
            ):
                reject(
                    token,
                    "oracle.escape",
                    "phase oracle cannot allocate, transfer ownership, declare Output or contain callbacks",
                )
            if token.text == "control":
                if statement.control_else is not None or len(statement.arguments) != 1:
                    reject(
                        token,
                        "oracle",
                        "phase oracle admits one predicate without else",
                    )
                actions = statement.control_body
                if (
                    actions is None
                    or len(actions) != 1
                    or actions[0].token.text != "phase"
                ):
                    reject(
                        token,
                        "oracle",
                        "borrowed predicate may control one phase effect only",
                    )
                condition = substitute(
                    closed(statement.arguments[0], variable, integers, width, record),
                    variable,
                    target,
                )
                phases = body(
                    actions, variable, target, active, integers, width, record
                )
                if len(phases[0].arguments) == 1:
                    # Reuse the admitted conditional real-phase path for phase(a).
                    one = Expression(
                        Token("+", token.location),
                        (
                            Expression(Token("1", token.location)),
                            Expression(Token("0", token.location)),
                        ),
                    )
                    phases = (
                        replace(phases[0], arguments=(one, phases[0].arguments[0])),
                    )
                result.append(
                    replace(statement, arguments=(condition,), control_body=phases)
                )
            elif token.text == "Z" and statement.control_body is None:
                if (
                    len(statement.arguments) != 1
                    or statement.arguments[0].token.text != "["
                ):
                    reject(token, "oracle", "borrowed Z requires one indexed array bit")
                selected = substitute(
                    closed(statement.arguments[0], variable, integers, width, record),
                    variable,
                    target,
                )
                result.append(
                    Statement(
                        Token("phase", token.location),
                        (selected, Expression(Token("pi", token.location))),
                    )
                )
            elif token.text == "phase" and statement.control_body is None:
                if not 1 <= len(statement.arguments) <= 2:
                    reject(
                        token, "oracle", "phase effect requires one or two arguments"
                    )
                arguments = tuple(
                    substitute(
                        closed(arg, variable, integers, width, record), variable, target
                    )
                    for arg in statement.arguments
                )
                result.append(replace(statement, arguments=arguments))
            elif token.text in oracles and statement.control_body is None:
                callee = oracles[token.text]
                port = borrowed(callee)
                position = callee.parameters.index(port)
                if (
                    len(statement.arguments) != len(callee.parameters)
                    or statement.arguments[position].children
                    or statement.arguments[position].token.text != variable
                ):
                    reject(
                        token,
                        "oracle.alias",
                        "phase callback must forward exactly its borrowed target; aliases and captures are rejected",
                    )
                arguments = tuple(
                    substitute(
                        closed(arg, variable, integers, width, record), variable, target
                    )
                    for arg in statement.arguments
                )
                result.extend(
                    invoke(token, arguments, active, width, frozenset(integers), record)
                )
            else:
                reject(
                    token,
                    "oracle.escape",
                    "borrowed phase oracle admits only phase effects and checked phase calls; no writes or Output escape",
                )
        if not statements:
            reject(target.token, "oracle", "phase callback must be nonempty")
        return tuple(result)

    def invoke(
        token: Token,
        arguments: tuple[Expression, ...],
        active: tuple[str, ...],
        width: int | None,
        symbolic: frozenset[str] = frozenset(),
        record: str | None = None,
    ) -> tuple[Statement, ...]:
        function = oracles.get(token.text)
        if function is None or token.text == "main":
            reject(
                token,
                "oracle",
                "Grover callback must reference a checked Const-array phase oracle",
            )
        if token.text in active:
            reject(token, "recursion", "phase oracle call graph must be acyclic")
        if len(active) >= 32:
            reject(token, "budget.oracle", "phase oracle call depth exceeds 32")
        port = borrowed(function)
        if port.record != record:
            reject(
                token,
                "oracle.alias",
                "phase oracle requires the matching nominal record layout",
            )
        if len(arguments) != len(function.parameters):
            reject(
                token,
                "oracle.alias",
                "phase oracle argument count must match; no alias or Output ports",
            )
        if port.width is not None and width != port.width:
            reject(token, "oracle.alias", "phase oracle array width must match exactly")
        target = arguments[function.parameters.index(port)]
        integers = {}
        for parameter, argument in zip(function.parameters, arguments, strict=True):
            if parameter is port:
                continue
            if not argument.children and argument.token.text in symbolic:
                integers[parameter.name.text] = argument
                continue
            negative = argument.token.text == "-" and len(argument.children) == 1
            magnitude = argument.children[0] if negative else argument
            if magnitude.children or not magnitude.token.text.isdecimal():
                reject(
                    argument.token,
                    "oracle.capture",
                    "oracle integer arguments require exact bound literals, without quantum captures",
                )
            checked_integer(
                (-1 if negative else 1) * int(magnitude.token.text), argument.token
            )
            integers[parameter.name.text] = argument
        return body(
            function.statements,
            port.name.text,
            target,
            (*active, token.text),
            integers,
            width,
            record,
        )

    # Validate every admitted borrowed definition, rather than erasing dead errors.
    for function in oracles.values():
        port = borrowed(function)
        symbolic = frozenset(
            p.name.text for p in function.parameters if p.role == "integer"
        )
        invoke(
            function.name,
            tuple(Expression(p.name) for p in function.parameters),
            (),
            port.width,
            symbolic,
            port.record,
        )

    def scope(
        statements: tuple[Statement, ...], parameters: Mapping[str, Parameter]
    ) -> tuple[Statement, ...]:
        result: list[Statement] = []
        for statement in statements:
            token, args = statement.token, statement.arguments
            if token.text == "grover_search":
                repetitions, oracle, target = args
                call = replace(
                    statement,
                    token=Token("grover_operator", token.location),
                    arguments=(
                        oracle,
                        Expression(Token("hadamard_transform", token.location)),
                        target,
                    ),
                )
                search_loop = Statement(
                    Token("power", token.location),
                    (repetitions,),
                    repeat_index=Token(
                        f"$grover_search{token.location.line}_{token.location.column}",
                        token.location,
                    ),
                    repeat_body=(call,),
                )
                result.extend(
                    scope(
                        (
                            Statement(
                                Token("hadamard_transform", token.location), (target,)
                            ),
                            search_loop,
                        ),
                        parameters,
                    )
                )
                continue
            if token.text == "grover_operator":
                oracle, preparation, target = args
                if (
                    preparation.token.text != "hadamard_transform"
                    or preparation.children
                ):
                    reject(
                        preparation.token,
                        "oracle",
                        "Grover preparation supports only the built-in hadamard_transform",
                    )
                port = parameters.get(target.token.text)
                if (
                    target.children
                    or port is None
                    or port.kind != "qbit[]"
                    or port.role == "const"
                ):
                    reject(
                        target.token,
                        "oracle.alias",
                        "Grover target must be one whole writable array, without slices, captures or alias ports",
                    )
                if statement.oracle_callback is None:
                    phases = invoke(
                        oracle.token, (target,), (), port.width, record=port.record
                    )
                else:
                    variable, actions = statement.oracle_callback
                    if (
                        (
                            variable.text in parameters
                            and variable.text != target.token.text
                        )
                        or variable.text in definitions
                        or variable.text in constants
                        or variable.text == "pi"
                    ):
                        reject(
                            variable,
                            "oracle.capture",
                            "callback parameter cannot shadow a bound name",
                        )
                    phases = body(
                        actions, variable.text, target, (), {}, port.width, port.record
                    )
                result.extend(phases)
                h = Statement(Token("hadamard_transform", token.location), (target,))
                if port.width is not None and 10 < port.width <= 12:
                    zero = Expression(
                        Token("~", token.location),
                        (
                            Expression(
                                Token("[", token.location),
                                (target, Expression(Token("0", token.location))),
                            ),
                        ),
                    )
                    for bit in range(1, port.width):
                        item = Expression(
                            Token("~", token.location),
                            (
                                Expression(
                                    Token("[", token.location),
                                    (
                                        target,
                                        Expression(Token(str(bit), token.location)),
                                    ),
                                ),
                            ),
                        )
                        zero = Expression(Token("&", token.location), (zero, item))
                    one = Expression(
                        Token("+", token.location),
                        (
                            Expression(Token("0", token.location)),
                            Expression(Token("1", token.location)),
                        ),
                    )
                    flip_zero = Statement(
                        Token("control", token.location),
                        (zero,),
                        control_body=(
                            Statement(
                                Token("phase", token.location),
                                (one, Expression(Token("pi", token.location))),
                            ),
                        ),
                    )
                    global_sign = Statement(
                        Token("phase", token.location),
                        (Expression(Token("pi", token.location)),),
                    )
                    result.extend((h, flip_zero, global_sign, h))
                    continue
                loop = Token(
                    f"$oracle_flip{token.location.line}_{token.location.column}",
                    token.location,
                )
                index = Expression(
                    Token("[", token.location), (target, Expression(loop))
                )
                flip = Statement(
                    Token("repeat", token.location),
                    (
                        (
                            Expression(Token(str(port.width), token.location))
                            if port.record is not None
                            else Expression(
                                Token(".", token.location),
                                (target, Expression(Token("len", token.location))),
                            )
                        ),
                    ),
                    repeat_index=loop,
                    repeat_body=(Statement(Token("X", token.location), (index,)),),
                )
                phase = Statement(
                    Token("phase", token.location),
                    (Expression(Token("pi", token.location)),),
                )
                marked = Statement(
                    Token("control", token.location), (target,), control_body=(phase,)
                )
                # -H X^n (I-2|1><1|) X^n H = 2|s><s|-I, including -1.
                result.extend((h, flip, marked, flip, phase, h))
            elif token.text in oracles:
                function = oracles[token.text]
                declared = borrowed(function)
                position = function.parameters.index(declared)
                if len(args) != len(function.parameters) or args[position].children:
                    reject(
                        token, "oracle.alias", "phase oracle requires one whole array"
                    )
                port = parameters.get(args[position].token.text)
                if port is None or port.kind != "qbit[]":
                    reject(
                        token, "oracle.alias", "phase oracle target must be an array"
                    )
                if declared.width is not None and port.width != declared.width:
                    reject(
                        token,
                        "oracle.alias",
                        "phase oracle array width must match exactly",
                    )
                resolved = tuple(
                    constants.get(arg.token.text, arg) if not arg.children else arg
                    for arg in args
                )
                result.extend(
                    invoke(token, resolved, (), port.width, record=port.record)
                )
            else:

                def optional(
                    value: tuple[Statement, ...] | None,
                ) -> tuple[Statement, ...] | None:
                    return None if value is None else scope(value, parameters)

                result.append(
                    replace(
                        statement,
                        within=optional(statement.within),
                        apply=scope(statement.apply, parameters),
                        control_body=optional(statement.control_body),
                        control_else=optional(statement.control_else),
                        repeat_body=scope(statement.repeat_body, parameters),
                        inverse_body=optional(statement.inverse_body),
                        static_if_body=optional(statement.static_if_body),
                    )
                )
        return tuple(result)

    normalized = {}
    for name, function in definitions.items():
        if name in oracles:
            continue
        parameters = {p.name.text: p for p in function.parameters}
        parameters.update(
            {
                s.declaration.name.text: s.declaration
                for s in function.statements
                if s.declaration is not None
            }
        )
        normalized[name] = replace(
            function, statements=scope(function.statements, parameters)
        )
    return normalized
