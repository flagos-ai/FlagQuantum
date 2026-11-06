"""Closed source callbacks specialized before first-order oracle adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .syntax import Expression, Function, Parameter, Statement, Token, reject

if TYPE_CHECKING:
    from .syntax import Parser

MAX_CALLBACK_STATEMENTS = 4096
MAX_CALLBACK_DEPTH = 16

BORROWED_EFFECT_SIGNATURES = {
    "qfunc(qbit[],qbit[])": ("qbit[]", "qbit[]"),
    "qfunc(qnum)": ("qnum",),
}
CALLBACK_ARRAY = "qfunc[]()"

QUANTUM_SIGNATURES = {
    "qfunc(qbit[])": ("qbit[]",),
    "qfunc(qnum,qbit)": ("qnum", "qbit"),
}


@dataclass(frozen=True)
class Lambda(Expression):
    parameters: tuple[Token, ...] = ()
    body: tuple[Statement, ...] = ()


def parse_signature(parser: Parser) -> str:
    token = parser.take("qfunc")
    if parser.current.text == "[":
        for part in ("[", "]", "(", ")"):
            parser.take(part)
        return CALLBACK_ARRAY
    parser.take("(")
    types = []
    while parser.current.text != ")":
        token = parser.current
        if token.text not in {"real", "qnum", "qbit"}:
            reject(
                token, "callback.signature", "callback requires borrowed quantum ports"
            )
        parser.take(token.text)
        kind = token.text
        if parser.current.text == "[":
            parser.take("[")
            parser.take("]")
            kind += "[]"
        types.append(kind)
        if parser.current.text != ",":
            break
        parser.take(",")
    parser.take(")")
    signature = "qfunc(" + ",".join(types) + ")"
    if signature not in QUANTUM_SIGNATURES and signature not in (
        BORROWED_EFFECT_SIGNATURES.keys() | {"qfunc(real,qbit[])"}
    ):
        reject(token, "callback.signature", "unsupported callback signature")
    return signature


def parse_lambda(parser: Parser, *, preparation: bool = False) -> Lambda:
    token = parser.take("lambda")
    if parser.callback_depth >= (2 if preparation else 1):
        reject(token, "callback.escape", "nested callback values are not admitted")
    parser.take("(")
    parameters = []
    if parser.current.text != ")":
        parameters.append(parser.name())
        while parser.current.text == ",":
            parser.take(",")
            parameters.append(parser.name())
    parser.take(")")
    parser.callback_depth += 1
    try:
        body = parser.block(allow_scopes=False, scope_depth=1)
    finally:
        parser.callback_depth -= 1
    return Lambda(token, parameters=tuple(parameters), body=body)


def specialize_quantum_callbacks(
    definitions: Mapping[str, Function], constants: Mapping[str, Expression]
) -> dict[str, Function]:
    wrappers = {
        name: f
        for name, f in definitions.items()
        if any(
            p.role == "callable" and p.kind in QUANTUM_SIGNATURES for p in f.parameters
        )
    }
    if not wrappers:
        for function in definitions.values():
            for p in function.parameters:
                if p.role == "const" and p.kind == "qnum" and p.width is None:
                    reject(
                        p.name,
                        "width",
                        "Const qnum requires an explicit unsigned width",
                    )
        return dict(definitions)
    count = 0
    types: dict[str, Parameter] = {}

    def expression(node: Expression, bindings: Mapping[str, Expression]) -> Expression:
        if not node.children:
            return bindings.get(node.token.text, node)
        return replace(
            node, children=tuple(expression(c, bindings) for c in node.children)
        )

    def closed(
        body: tuple[Statement, ...],
        ports: tuple[Token, ...],
        kinds: tuple[str, ...],
        stack: tuple[str, ...] = (),
    ) -> None:
        if any(p.text in definitions or p.text in constants for p in ports):
            reject(
                ports[0],
                "callback.capture",
                "callback parameters cannot shadow definitions or constants",
            )
        formal_kinds = dict(zip((p.text for p in ports), kinds, strict=True))
        allowed = {p.text for p in ports} | set(constants) | {"pi"}
        if not body or len({p.text for p in ports}) != len(ports):
            reject(
                ports[0] if ports else next(iter(definitions.values())).name,
                "callback.signature",
                "callback must be nonempty with distinct parameters",
            )

        def check(node: Expression) -> None:
            if (
                not node.children
                and node.token.text.isidentifier()
                and node.token.text not in allowed
            ):
                reject(
                    node.token,
                    "callback.capture",
                    "callback cannot capture or escape a quantum value",
                )
            for child in node.children:
                check(child)

        for s in body:
            if s.declaration is not None or any(
                (
                    s.within is not None,
                    s.control_body is not None,
                    s.inverse_body is not None,
                    s.repeat_index is not None,
                )
            ):
                reject(
                    s.token,
                    "callback.escape",
                    "callback cannot allocate, declare Output or change ownership",
                )
            if s.token.text not in {"phase_oracle", "hadamard_transform"}:
                callee = definitions.get(s.token.text)
                if callee is None or any(
                    p.role not in {"inout", "const"} for p in callee.parameters
                ):
                    reject(
                        s.token,
                        "callback.escape",
                        "callback must forward borrowed quantum arguments only",
                    )
                if len(s.arguments) != len(callee.parameters) or any(
                    arg.children or formal_kinds.get(arg.token.text) != p.kind
                    for arg, p in zip(s.arguments, callee.parameters, strict=True)
                ):
                    reject(
                        s.token,
                        "callback.signature",
                        "callback must forward matching quantum kinds",
                    )
                if callee.name.text in stack:
                    reject(
                        s.token,
                        "callback.recursion",
                        "callback call graph must be acyclic",
                    )
                # Existing oracle validators check the predicate body independently.
            for arg in (
                s.arguments[1:] if s.token.text == "phase_oracle" else s.arguments
            ):
                check(arg)
            if s.predicate_callback is not None or s.oracle_callback is not None:
                reject(s.token, "callback.escape", "callback nesting is not admitted")

    def validate_wrapper(function: Function) -> None:
        names = {p.name.text for p in function.parameters}
        if len(names) != len(function.parameters) or any(
            p.role not in {"callable", "inout"} for p in function.parameters
        ):
            reject(
                function.name,
                "callback.signature",
                "quantum wrappers require distinct borrowed ports",
            )

        ports = {p.name.text: p for p in function.parameters}
        if not function.statements:
            reject(
                function.name, "callback.signature", "quantum wrapper must be nonempty"
            )

        def check_argument(node: Expression, allowed: set[str]) -> None:
            if isinstance(node, Lambda):
                reject(
                    node.token,
                    "callback.escape",
                    "nested callback values are not admitted",
                )
            if (
                not node.children
                and node.token.text.isidentifier()
                and node.token.text not in allowed | set(constants) | {"pi"}
            ):
                reject(
                    node.token,
                    "callback.capture",
                    "wrapper cannot capture a quantum value",
                )
            for child in node.children:
                check_argument(child, allowed)

        def walk(body: tuple[Statement, ...]) -> None:
            for s in body:
                for arg in (
                    s.arguments[1:] if s.token.text == "phase_oracle" else s.arguments
                ):
                    check_argument(arg, names)
                if s.token.text in ports:
                    port = ports[s.token.text]
                    kinds = QUANTUM_SIGNATURES.get(port.kind)
                    if (
                        kinds is None
                        or len(s.arguments) != len(kinds)
                        or any(
                            arg.children
                            or ports.get(arg.token.text) is None
                            or ports[arg.token.text].kind != kind
                            for arg, kind in zip(s.arguments, kinds or (), strict=False)
                        )
                        or len({arg.token.text for arg in s.arguments})
                        != len(s.arguments)
                    ):
                        reject(
                            s.token,
                            "callback.signature",
                            "callback invocation arity disagrees",
                        )
                if s.token.text == "hadamard_transform" and (
                    len(s.arguments) != 1
                    or s.arguments[0].children
                    or s.arguments[0].token.text not in ports
                    or ports[s.arguments[0].token.text].role != "inout"
                ):
                    reject(
                        s.token,
                        "callback.signature",
                        "Hadamard requires one whole borrowed quantum port",
                    )
                if s.token.text == "phase_oracle" and s.predicate_callback is None:
                    if (
                        len(s.arguments) != 2
                        or s.arguments[0].children
                        or s.arguments[0].token.text not in definitions
                    ):
                        reject(
                            s.token,
                            "callback.signature",
                            "wrapper phase oracle requires a known predicate",
                        )
                if s.predicate_callback is not None:
                    a, b, actions = s.predicate_callback
                    if (
                        a.text == b.text
                        or len(actions) != 1
                        or actions[0].token.text not in ports
                    ):
                        reject(
                            s.token,
                            "callback.signature",
                            "predicate callback must forward one borrowed callback",
                        )
                    invoked = ports[actions[0].token.text]
                    if (
                        invoked.kind != "qfunc(qnum,qbit)"
                        or len(actions[0].arguments) != 2
                    ):
                        reject(
                            s.token,
                            "callback.signature",
                            "predicate callback signature disagrees",
                        )
                    if tuple(arg.token.text for arg in actions[0].arguments) != (
                        a.text,
                        b.text,
                    ) or any(arg.children for arg in actions[0].arguments):
                        reject(
                            s.token,
                            "callback.alias",
                            "predicate callback must forward distinct input and marker",
                        )
                    for arg in actions[0].arguments:
                        check_argument(arg, {a.text, b.text})
                if s.within is not None:
                    if not s.within or any(
                        p.token.text != "hadamard_transform" for p in s.within
                    ):
                        reject(
                            s.token,
                            "callback.scope",
                            "callback preparation admits only borrowed Hadamard transforms",
                        )
                    walk(s.within)
                    walk(s.apply)
                elif (
                    s.token.text
                    not in names
                    | {"phase_oracle", "hadamard_transform"}
                    | wrappers.keys()
                ):
                    reject(
                        s.token,
                        "callback.escape",
                        "quantum wrapper cannot allocate or escape borrowed ports",
                    )
                if (
                    s.declaration is not None
                    or s.control_body is not None
                    or s.inverse_body is not None
                    or s.repeat_index is not None
                ):
                    reject(
                        s.token,
                        "callback.escape",
                        "quantum wrapper requires static borrowed composition",
                    )

        walk(function.statements)

    graph_steps = 0

    def wrapper_graph(name: str, stack: tuple[str, ...]) -> None:
        nonlocal graph_steps
        graph_steps += 1
        if graph_steps > MAX_CALLBACK_STATEMENTS:
            reject(
                wrappers[name].name,
                "budget.callback",
                "unused wrapper expansion exceeds 4096 calls",
            )
        if name in stack or len(stack) >= MAX_CALLBACK_DEPTH:
            reject(
                wrappers[name].name,
                "callback.recursion",
                "wrapper graph must be acyclic and bounded",
            )
        pending = list(wrappers[name].statements)
        while pending:
            statement = pending.pop()
            if statement.token.text in wrappers:
                wrapper_graph(statement.token.text, (*stack, name))
            pending.extend(statement.within or ())
            pending.extend(statement.apply)

    for name, function in wrappers.items():
        validate_wrapper(function)
        wrapper_graph(name, ())

    def scope(
        body: tuple[Statement, ...],
        bindings: Mapping[str, Expression],
        callbacks: Mapping[str, Lambda],
        stack: tuple[str, ...],
    ) -> tuple[Statement, ...]:
        nonlocal count
        result: list[Statement] = []
        for s in body:
            count += 1
            if count > MAX_CALLBACK_STATEMENTS or len(stack) > MAX_CALLBACK_DEPTH:
                reject(
                    s.token,
                    "budget.callback",
                    "callback expansion exceeds 4096 statements or sixteen levels",
                )
            args = tuple(expression(a, bindings) for a in s.arguments)
            name = s.token.text
            if name in wrappers:
                f = wrappers[name]
                if name in stack:
                    reject(
                        s.token,
                        "callback.recursion",
                        "callback expansion must be acyclic",
                    )
                if len(args) != len(f.parameters):
                    reject(
                        s.token,
                        "callback.signature",
                        "wrapper argument count disagrees",
                    )
                env: dict[str, Expression] = {}
                bound: dict[str, Lambda] = {}
                quantum = []
                for p, arg in zip(f.parameters, args, strict=True):
                    if p.role != "callable":
                        if arg.children or isinstance(arg, Lambda):
                            reject(
                                arg.token,
                                "callback.alias",
                                "wrapper requires whole distinct borrowed arguments",
                            )
                        actual = types.get(arg.token.text)
                        if (
                            actual is None
                            or actual.kind != p.kind
                            or (p.width is not None and p.width != actual.width)
                        ):
                            reject(
                                arg.token,
                                "callback.signature",
                                "wrapper borrowed kind or width disagrees",
                            )
                        quantum.append(arg.token.text)
                        env[p.name.text] = arg
                        continue
                    kinds = QUANTUM_SIGNATURES.get(p.kind)
                    if kinds is None:
                        reject(
                            p.name,
                            "callback.signature",
                            "mixed classical callback profiles are not admitted",
                        )
                    callback = (
                        arg
                        if isinstance(arg, Lambda)
                        else callbacks.get(arg.token.text)
                    )
                    if callback is None:
                        named = definitions.get(arg.token.text)
                        if (
                            named is None
                            or tuple(q.kind for q in named.parameters) != kinds
                            or any(
                                q.role not in {"inout", "const"}
                                for q in named.parameters
                            )
                        ):
                            reject(
                                arg.token,
                                "callback.signature",
                                "callback signature or borrowed roles disagree",
                            )
                        callback = Lambda(
                            arg.token,
                            parameters=tuple(q.name for q in named.parameters),
                            body=(
                                Statement(
                                    named.name,
                                    tuple(Expression(q.name) for q in named.parameters),
                                ),
                            ),
                        )
                    if len(callback.parameters) != len(kinds):
                        reject(
                            arg.token,
                            "callback.signature",
                            "callback argument count disagrees",
                        )
                    closed(callback.body, callback.parameters, kinds)
                    bound[p.name.text] = callback
                if len(set(quantum)) != len(quantum):
                    reject(
                        s.token,
                        "callback.alias",
                        "wrapper quantum arguments must be disjoint",
                    )
                result.extend(scope(f.statements, env, bound, (*stack, name)))
            elif name in callbacks:
                cb = callbacks[name]
                if (
                    len(args) != len(cb.parameters)
                    or len({a.token.text for a in args}) != len(args)
                    or any(a.children for a in args)
                ):
                    reject(
                        s.token,
                        "callback.alias",
                        "callback requires distinct whole arguments and matching arity",
                    )
                result.extend(
                    scope(
                        cb.body,
                        dict(zip((p.text for p in cb.parameters), args, strict=True)),
                        {},
                        (*stack, name),
                    )
                )
            elif s.within is not None and stack:
                prepared = scope(s.within, bindings, callbacks, stack)
                result.extend(
                    (
                        *prepared,
                        *scope(s.apply, bindings, callbacks, stack),
                        *reversed(prepared),
                    )
                )
            else:
                pred = s.predicate_callback
                if pred is not None and callbacks:
                    a, b, actions = pred
                    pred = (a, b, scope(actions, {}, callbacks, stack))

                def optional(
                    value: tuple[Statement, ...] | None,
                ) -> tuple[Statement, ...] | None:
                    return (
                        None
                        if value is None
                        else scope(value, bindings, callbacks, stack)
                    )

                result.append(
                    replace(
                        s,
                        arguments=args,
                        predicate_callback=pred,
                        qpe_body=optional(s.qpe_body),
                        within=optional(s.within),
                        apply=scope(s.apply, bindings, callbacks, stack),
                        control_body=optional(s.control_body),
                        control_else=optional(s.control_else),
                        inverse_body=optional(s.inverse_body),
                        repeat_body=scope(s.repeat_body, bindings, callbacks, stack),
                        static_if_body=optional(s.static_if_body),
                    )
                )
        return tuple(result)

    normalized = {}
    for name, f in definitions.items():
        if name in wrappers:
            continue
        types = {p.name.text: p for p in f.parameters}
        for statement in f.statements:
            if statement.token.text == "allocate" and len(statement.arguments) == 2:
                width, target = statement.arguments
                port = types.get(target.token.text)
                if (
                    port is not None
                    and port.width is None
                    and not width.children
                    and width.token.text.isdecimal()
                ):
                    types[target.token.text] = replace(
                        port, width=int(width.token.text)
                    )
        normalized[name] = replace(
            f,
            parameters=tuple(types[p.name.text] for p in f.parameters),
            statements=scope(f.statements, {}, {}, ()),
        )
    return normalized
