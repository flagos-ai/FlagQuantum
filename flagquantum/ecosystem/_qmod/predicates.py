"""Static predicate borrowing and XOR-to-phase composition at the Qmod boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .integers import checked_integer
from .permutations import is_const_cx_permutation
from .selector_rotations import is_selector_helper
from .syntax import Expression, Function, Parameter, Record, Statement, Token, reject

if TYPE_CHECKING:
    from .syntax import Parser


def parse_phase_oracle_call(parser: Parser, token: Token, depth: int) -> Statement:
    """Read a named predicate or a closed two-argument forwarding lambda."""
    parser.take("(")
    callback = None
    if parser.current.text == "lambda":
        marker = parser.take("lambda")
        parser.take("(")
        read = parser.name()
        parser.take(",")
        flag = parser.name()
        parser.take(")")
        callback = (read, flag, parser.block(allow_scopes=False, scope_depth=depth + 1))
        predicate = Expression(marker)
    else:
        predicate = Expression(parser.name())
    parser.take(",")
    target = parser.call_argument()
    parser.take(")")
    parser.take(";")
    return Statement(token, (predicate, target), predicate_callback=callback)


@dataclass(frozen=True)
class Predicate:
    function: Function
    read: Parameter
    flag: Parameter
    expression: Expression


def expand_predicate_oracles(
    definitions: Mapping[str, Function],
    constants: Mapping[str, Expression],
    records: Mapping[str, Record],
) -> dict[str, Function]:
    """Validate restricted XOR predicates, then erase their static callback syntax."""
    predicates: dict[str, Predicate] = {}
    for name, function in definitions.items():
        if is_const_cx_permutation(function) or is_selector_helper(function):
            continue
        reads = [
            p
            for p in function.parameters
            if p.role == "const" and p.kind in {"qbit[]", "qnum"}
        ]
        writes = [p for p in function.parameters if p.role == "inout"]
        if not reads or not writes:
            continue
        if (
            not function.permutation
            or len(reads) != 1
            or len(writes) != 1
            or any(
                p not in reads + writes
                and not (p.role == "integer" and p.kind == "int")
                for p in function.parameters
            )
            or len({p.name.text for p in function.parameters})
            != len(function.parameters)
        ):
            reject(
                function.name,
                "predicate",
                "predicate requires one Const input, one distinct writable marker and scalar integer ports",
            )
        read, flag = reads[0], writes[0]
        if (
            flag.kind not in {"qbit", "qnum"}
            or flag.width not in {None, 1}
            or flag.record is not None
        ):
            reject(
                flag.name,
                "predicate",
                "predicate marker must be exactly one qbit or one-bit qnum",
            )
        if len(function.statements) != 1:
            reject(
                function.name,
                "predicate.escape",
                "predicate body must be exactly marker ^= expression; no Output or ownership escape",
            )
        assignment = function.statements[0]
        if (
            assignment.token.text != "^="
            or assignment.arguments[0].children
            or assignment.arguments[0].token.text != flag.name.text
        ):
            reject(
                function.name,
                "predicate.escape",
                "predicate may only XOR its own marker; no allocation, mutation or Output escape",
            )
        predicates[name] = Predicate(function, read, flag, assignment.arguments[1])

    # A numeric Const width may be inferred only for a checked borrowed predicate.
    for name, function in definitions.items():
        if name in predicates:
            continue
        for port in function.parameters:
            if port.role == "const" and port.kind == "qnum" and port.width is None:
                reject(
                    port.name, "width", "Const qnum requires an explicit unsigned width"
                )

    def expression(
        node: Expression,
        predicate: Predicate,
        integers: Mapping[str, Expression],
        target: Expression,
    ) -> Expression:
        read = predicate.read
        if not node.children:
            text = node.token.text
            if text == predicate.flag.name.text:
                reject(
                    node.token,
                    "predicate.alias",
                    "predicate expression cannot read or capture its marker",
                )
            if text == read.name.text:
                return target
            if text in integers:
                return integers[text]
            if text in constants:
                return constants[text]
            if not text.isdecimal():
                reject(
                    node.token,
                    "predicate.capture",
                    "predicate expression requires bounded integers and its borrowed input only",
                )
            return node
        if node.token.text == ".":
            root, field = node.children
            if (
                read.record is None
                or root.children
                or root.token.text != read.name.text
                or field.children
            ):
                reject(
                    node.token,
                    "predicate.alias",
                    "field must belong to the declared borrowed record",
                )
            layout = records[read.record]
            declared = next(
                (p for p in layout.fields if p.name.text == field.token.text), None
            )
            if (
                declared is None
                or declared.kind not in {"qbit", "qnum"}
                or declared.width is None
            ):
                reject(
                    field.token,
                    "predicate.type",
                    "predicate record fields require fixed unsigned scalar values",
                )
            return replace(node, children=(target, field))
        if node.token.text not in {
            "[",
            "+",
            "-",
            "*",
            "**",
            "%",
            "min",
            "max",
            "&",
            "|",
            "^",
            "~",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            reject(node.token, "predicate.type", "unsupported predicate arithmetic")
        if node.token.text == "[" and (
            read.record is not None
            or node.children[0].children
            or node.children[0].token.text != read.name.text
        ):
            reject(
                node.token,
                "predicate.alias",
                "indexed reads require the whole borrowed array",
            )
        return replace(
            node,
            children=tuple(
                expression(child, predicate, integers, target)
                for child in node.children
            ),
        )

    def one_bit(node: Expression, predicate: Predicate) -> bool:
        if node.token.text in {"==", "!=", "<", "<=", ">", ">="}:
            return True
        if node.token.text in {"&", "|", "^", "*"}:
            return all(one_bit(child, predicate) for child in node.children)
        if node.token.text == "~":
            return one_bit(node.children[0], predicate)
        if node.token.text == "[":
            return True
        if node.token.text == "." and predicate.read.record is not None:
            return any(
                p.name.text == node.children[1].token.text and p.width == 1
                for p in records[predicate.read.record].fields
            )
        return not node.children and node.token.text in {"0", "1"}

    # Check unused definitions too; symbolic integers are resolved at invocation.
    for predicate in predicates.values():
        integers = {
            p.name.text: Expression(p.name)
            for p in predicate.function.parameters
            if p.role == "integer"
        }
        expression(
            predicate.expression, predicate, integers, Expression(predicate.read.name)
        )
        if not one_bit(predicate.expression, predicate):
            reject(
                predicate.function.name,
                "predicate.type",
                "predicate XOR expression must prove a one-bit Boolean result",
            )

    def specialize(
        token: Token,
        arguments: tuple[Expression, ...],
        target: Expression,
        marker: Expression,
        actual: Parameter,
        *,
        borrow_array_word: bool = False,
    ) -> Expression:
        predicate = predicates.get(token.text)
        if predicate is None:
            reject(
                token,
                "predicate",
                "phase_oracle requires a checked XOR qperm predicate",
            )
        ports = predicate.function.parameters
        if len(arguments) != len(ports):
            reject(
                token, "predicate.alias", "predicate argument count must match exactly"
            )
        read_arg, flag_arg = (
            arguments[ports.index(predicate.read)],
            arguments[ports.index(predicate.flag)],
        )
        if (
            read_arg.children
            or flag_arg.children
            or read_arg.token.text != target.token.text
            or flag_arg.token.text != marker.token.text
            or target.children
            or marker.children
            or target.token.text == marker.token.text
        ):
            reject(
                token,
                "predicate.alias",
                "callback must forward its distinct whole input and marker without capture or aliases",
            )
        array_word = (
            borrow_array_word
            and actual.kind == "qbit[]"
            and predicate.read.kind == "qnum"
            and actual.record is None
            and predicate.read.record is None
        )
        if (
            (actual.kind != predicate.read.kind and not array_word)
            or actual.record != predicate.read.record
            or (
                predicate.read.width is not None
                and actual.width != predicate.read.width
            )
        ):
            reject(
                token,
                "predicate.type",
                "borrowed predicate input requires its matching nominal record or array width",
            )
        integers = {}
        for port, argument in zip(ports, arguments, strict=True):
            if port.role != "integer":
                continue
            value = (
                constants.get(argument.token.text, argument)
                if not argument.children
                else argument
            )
            negative = value.token.text == "-" and len(value.children) == 1
            magnitude = value.children[0] if negative else value
            if magnitude.children or not magnitude.token.text.isdecimal():
                reject(
                    argument.token,
                    "predicate.capture",
                    "predicate integer specialization requires exact literals",
                )
            checked_integer(
                (-1 if negative else 1) * int(magnitude.token.text), argument.token
            )
            integers[port.name.text] = value
        word = (
            Expression(Token("$predicate_word", target.token.location), (target,))
            if array_word
            else target
        )
        return expression(predicate.expression, predicate, integers, word)

    def phase_statement(
        token: Token, condition: Expression, value: Expression
    ) -> Statement:
        if not condition.children and condition.token.text in {"0", "1"}:
            arguments: tuple[Expression, ...]
            if condition.token.text == "0":
                arguments = (Expression(Token("0", token.location)),)
            elif not value.children and value.token.text == "1":
                arguments = (Expression(Token("pi", token.location)),)
            else:
                arguments = (value, Expression(Token("pi", token.location)))
            return Statement(Token("phase", token.location), arguments)
        weighted = Expression(
            Token("+", token.location), (Expression(Token("0", token.location)), value)
        )
        phase = Statement(
            Token("phase", token.location),
            (weighted, Expression(Token("pi", token.location))),
        )
        return Statement(
            Token("control", token.location), (condition,), control_body=(phase,)
        )

    count = 0

    def bind_input(node: Expression, formal: Token, target: Expression) -> Expression:
        if not node.children and node.token.text == formal.text:
            return target
        if node.token.text == ".":
            return replace(
                node,
                children=(
                    bind_input(node.children[0], formal, target),
                    node.children[1],
                ),
            )
        return replace(
            node,
            children=tuple(
                bind_input(child, formal, target) for child in node.children
            ),
        )

    def scope(
        statements: tuple[Statement, ...], parameters: Mapping[str, Parameter]
    ) -> tuple[Statement, ...]:
        nonlocal count
        result: list[Statement] = []
        for statement in statements:
            count += 1
            if count > 4096:
                reject(
                    statement.token,
                    "budget.predicate",
                    "predicate expansion exceeds 4096 statements",
                )
            token, args = statement.token, statement.arguments
            if token.text == "phase_oracle":
                reference, target = args
                actual = parameters.get(target.token.text)
                if target.children or actual is None:
                    reject(
                        target.token,
                        "predicate.alias",
                        "phase_oracle requires a declared whole input",
                    )
                if statement.predicate_callback is None:
                    marker = Expression(Token("$predicate_marker", token.location))
                    predicate = predicates.get(reference.token.text)
                    if predicate is None or len(predicate.function.parameters) != 2:
                        reject(
                            reference.token,
                            "predicate",
                            "named phase_oracle requires one input and one marker",
                        )
                    supplied = tuple(
                        target if p is predicate.read else marker
                        for p in predicate.function.parameters
                    )
                    condition = specialize(
                        reference.token, supplied, target, marker, actual
                    )
                else:
                    read, flag, actions = statement.predicate_callback
                    if (
                        read.text == flag.text
                        or len(actions) != 1
                        or actions[0].token.text not in predicates
                    ):
                        reject(
                            token,
                            "predicate.escape",
                            "predicate lambda must forward distinct input/marker formals in one checked call",
                        )
                    condition = specialize(
                        actions[0].token,
                        actions[0].arguments,
                        Expression(read),
                        Expression(flag),
                        actual,
                    )

                    # Only the lexical input formal is substituted, never other captures.
                    condition = bind_input(condition, read, target)
                result.append(
                    phase_statement(
                        token, condition, Expression(Token("1", token.location))
                    )
                )
            elif token.text in predicates:
                predicate = predicates[token.text]
                ports = predicate.function.parameters
                if len(args) != len(ports):
                    reject(
                        token,
                        "predicate.alias",
                        "predicate argument count must match exactly",
                    )
                target, marker = (
                    args[ports.index(predicate.read)],
                    args[ports.index(predicate.flag)],
                )
                actual, marker_port = parameters.get(target.token.text), parameters.get(
                    marker.token.text
                )
                if (
                    actual is None
                    or marker_port is None
                    or marker_port.kind not in {"qbit", "qnum"}
                    or marker_port.width != 1
                    or marker_port.role == "const"
                ):
                    reject(
                        token,
                        "predicate.alias",
                        "direct predicate requires a distinct writable initialized one-bit marker",
                    )
                condition = specialize(
                    token, args, target, marker, actual, borrow_array_word=True
                )
                h = Statement(Token("hadamard_transform", token.location), (marker,))
                result.extend((h, phase_statement(token, condition, marker), h))
            else:
                callback = statement.oracle_callback
                if callback is not None:
                    formal, actions = callback
                    target = args[-1]
                    actual = parameters.get(target.token.text)
                    if actual is None or target.children:
                        reject(
                            token,
                            "predicate.alias",
                            "Grover callback requires a whole declared target",
                        )
                    callback = (
                        formal,
                        scope(
                            actions,
                            {formal.text: replace(actual, name=formal, role="const")},
                        ),
                    )

                def optional(
                    value: tuple[Statement, ...] | None,
                ) -> tuple[Statement, ...] | None:
                    return None if value is None else scope(value, parameters)

                result.append(
                    replace(
                        statement,
                        oracle_callback=callback,
                        qpe_body=optional(statement.qpe_body),
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
        if name in predicates:
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
