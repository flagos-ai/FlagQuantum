"""Check finite reversible Qmod control and QROM tables."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from .basis_regions import _name
from .quantum_callbacks import Lambda
from .syntax import Expression, Function, Statement, Token, reject


def _plain(step: Statement) -> bool:
    return step == Statement(step.token, step.arguments)


def _atom(node: Expression, text: str) -> bool:
    return not node.children and node.token.text == text


def _symbols(node: Expression) -> set[str]:
    if node.token.text == ".":
        return {_name(node)}
    if not node.children:
        return {node.token.text}
    return set().union(*(_symbols(child) for child in node.children))


def _actions(body: tuple[Statement, ...]) -> Iterator[Statement]:
    for step in body:
        yield step
        for part in (
            step.within,
            step.apply,
            step.repeat_body,
            step.inverse_body,
            step.static_if_body,
            step.control_body,
            step.control_else,
        ):
            if part:
                yield from _actions(part)
        for arg in step.arguments:
            if isinstance(arg, Lambda):
                yield from _actions(arg.body)


def _require(condition: object, token: Token, message: str) -> None:
    if not condition:
        reject(token, "zero_walk", message)


class _Row:
    """Execute checked computational controls without expanding quantum data."""

    def __init__(
        self, values: Mapping[str, int], widths: Mapping[str, int] | None = None
    ) -> None:
        self.widths = dict(widths or {})
        self.words = dict(values)
        self.locals: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.calls: list[tuple[str, bool]] = []

    def integer(self, node: Expression) -> int:
        token, args = node.token, node.children
        op = token.text
        if op in {"[", "."} or not args and op in self.words:
            return self.read(node)
        if not args:
            if op in {"True", "False"}:
                return int(op == "True")
            if op.isdecimal():
                return int(op)
        if len(args) == 2:
            a, b = (self.integer(item) for item in args)
            if op in {"+", "-", "*"}:
                return {"+": a + b, "-": a - b, "*": a * b}[op]
            if op in {"<", "<=", ">", ">=", "==", "!="}:
                return int(
                    {
                        "<": a < b,
                        "<=": a <= b,
                        ">": a > b,
                        ">=": a >= b,
                        "==": a == b,
                        "!=": a != b,
                    }[op]
                )
        reject(token, "zero_walk", "unsupported static zero-walk expression")

    def locate(self, node: Expression) -> tuple[str, int | None]:
        if node.token.text == "[" and len(node.children) == 2:
            base, index = node.children
            label = _name(base)
            offset = self.integer(index)
            label = self.aliases.get(label, label)
            _require(
                0 <= offset < self.widths.get(label, 32),
                index.token,
                "control index outside bounded proof",
            )
            return label, offset
        label = _name(node)
        return self.aliases.get(label, label), None

    def read(self, node: Expression) -> int:
        label, offset = self.locate(node)
        _require(label in self.words, node.token, "read of an uninitialized proof view")
        value = self.words[label]
        return value if offset is None else (value >> offset) & 1

    def toggle(
        self, node: Expression, value: int, *, packed_array: bool = False
    ) -> None:
        label, offset = self.locate(node)
        _require(
            label in self.words, node.token, "write to an uninitialized proof view"
        )
        shift = (4 if packed_array else 1) * (offset or 0)
        self.words[label] ^= value << shift

    def run(
        self,
        body: tuple[Statement, ...],
        *,
        allowed: Mapping[str, tuple[str, ...]] | None = None,
        qrom: str = "",
        wrapper: str = "",
        reverse: bool = False,
    ) -> None:
        for step in reversed(body) if reverse else body:
            token, args = step.token, step.arguments
            op = token.text
            if step.declaration is not None:
                _require(
                    step.declaration.kind == "qbit",
                    token,
                    "control work must be a scalar",
                )
                continue
            if step.within is not None:
                self.run(step.within, allowed=allowed, qrom=qrom, wrapper=wrapper)
                self.run(
                    step.apply,
                    allowed=allowed,
                    qrom=qrom,
                    wrapper=wrapper,
                    reverse=reverse,
                )
                self.run(
                    step.within,
                    allowed=allowed,
                    qrom=qrom,
                    wrapper=wrapper,
                    reverse=True,
                )
                continue
            if (
                op == wrapper
                and wrapper
                and len(args) == 2
                and isinstance(args[1], Lambda)
            ):
                callback = args[1]
                _require(
                    len(callback.parameters) == 1,
                    token,
                    "QROM view needs one bound alias",
                )
                label, alias = _name(args[0]), callback.parameters[0].text
                _require(
                    label in self.words
                    and alias not in self.words
                    and alias not in self.aliases,
                    token,
                    "QROM aliases must be fresh",
                )
                self.aliases[alias] = label
                self.run(
                    callback.body,
                    allowed=allowed,
                    qrom=qrom,
                    wrapper=wrapper,
                    reverse=reverse,
                )
                del self.aliases[alias]
                continue
            _require(
                _plain(step),
                token,
                "unsupported coherent operation in a zero-walk table",
            )
            if op in {"allocate", "free"} and len(args) == 1:
                label = _name(args[0])
                if (op == "allocate") != reverse:
                    _require(
                        label not in self.words, token, "control work must be fresh"
                    )
                    self.locals.add(label)
                    self.words[label] = 0
                else:
                    _require(
                        label in self.locals and self.words[label] == 0,
                        token,
                        "control work must be zero on release",
                    )
                    del self.words[label]
                    self.locals.remove(label)
            elif op == "X" and len(args) == 1:
                self.toggle(args[0], 1)
            elif op == "CX" and len(args) == 2:
                _require(
                    self.locate(args[0]) != self.locate(args[1]), token, "aliased CX"
                )
                self.toggle(args[1], self.read(args[0]))
            elif op == "CCX" and len(args) == 2 and args[0].token.text == "{":
                inputs = args[0].children
                _require(
                    len(inputs) == 2
                    and len({self.locate(arg) for arg in (*inputs, args[1])}) == 3,
                    token,
                    "aliased or malformed CCX",
                )
                self.toggle(args[1], self.read(inputs[0]) & self.read(inputs[1]))
            elif op == qrom and qrom and len(args) == 3 and args[2].token.text == "[":
                flag, target, data = args
                _require(
                    1 <= len(data.children) <= 18
                    and all(
                        item.token.text in {"True", "False"} and not item.children
                        for item in data.children
                    ),
                    token,
                    "QROM rows need bounded Boolean literals",
                )
                word = sum(
                    int(item.token.text == "True") << bit
                    for bit, item in enumerate(data.children)
                )
                self.toggle(
                    target,
                    self.read(flag) * word,
                    packed_array=self.locate(target)[1] is not None,
                )
            elif allowed is not None and op in allowed and not reverse:
                expected = allowed[op]
                if expected[0] == "$control":
                    _require(
                        len(args) == 2 and _atom(args[1], expected[1]),
                        token,
                        "reflection borrowed a different state",
                    )
                    self.calls.append((op, not bool(self.read(args[0]))))
                else:
                    _require(
                        tuple(_name(arg) for arg in args) == expected,
                        token,
                        "selection borrowed different outputs",
                    )
                    self.calls.append((op, True))
            else:
                reject(
                    token,
                    "zero_walk",
                    "unproved action in zero-walk control or QROM table",
                )


def _check_array_wrapper(function: Function) -> None:
    ports = function.parameters
    _require(
        len(ports) == 2
        and ports[0].kind == "qbit[]"
        and ports[1].kind == "qfunc(qbit[])"
        and len(function.statements) == 1,
        function.name,
        "array cast must only forward a borrowed array",
    )
    [call] = function.statements
    _require(
        _plain(call)
        and call.token.text == ports[1].name.text
        and len(call.arguments) == 1
        and _atom(call.arguments[0], ports[0].name.text),
        call.token,
        "array cast may only forward its view",
    )


def _check_qrom_writer(function: Function) -> None:
    ports = function.parameters
    _require(
        len(ports) == 3
        and ports[0].kind == "qbit"
        and ports[1].kind == "qbit[]"
        and ports[2].kind == "bool[]"
        and len(function.statements) == 1,
        function.name,
        "QROM writer must use a scalar flag and a static Boolean row",
    )
    [loop] = function.statements
    names = [port.name.text for port in ports]
    index = loop.repeat_index
    _require(
        index is not None
        and len(loop.arguments) == 1
        and _name(loop.arguments[0]) == names[2] + ".len"
        and len(loop.repeat_body) == 1,
        loop.token,
        "QROM writer must iterate its Boolean row",
    )
    assert index is not None
    [test] = loop.repeat_body
    _require(
        len(test.arguments) == 1
        and test.arguments[0].token.text == "["
        and [_name(item) for item in test.arguments[0].children]
        == [names[2], index.text]
        and test.static_if_body is not None
        and len(test.static_if_body) == 1,
        test.token,
        "QROM writer must guard each XOR with the corresponding Boolean literal",
    )
    assert test.static_if_body is not None
    [gate] = test.static_if_body
    _require(
        _plain(gate)
        and gate.token.text == "CX"
        and len(gate.arguments) == 2
        and _atom(gate.arguments[0], names[0])
        and gate.arguments[1].token.text == "["
        and [_name(item) for item in gate.arguments[1].children]
        == [names[1], index.text],
        gate.token,
        "QROM writer may only XOR its corresponding output bit",
    )


def _qrom_calls(function: Function, definitions: Mapping[str, Function]) -> str:
    calls = {
        step.token.text
        for step in _actions(function.statements)
        if step.token.text in definitions
        and len(step.arguments) == 3
        and step.arguments[-1].token.text == "["
    }
    _require(
        len(calls) == 1, function.name, "expanded QROM must use one checked writer"
    )
    [writer] = calls
    _check_qrom_writer(definitions[writer])
    return writer


def _check_qrom_body(
    function: Function,
    initial: Mapping[str, int],
    selectors: Mapping[str, int],
    definitions: Mapping[str, Function],
    *,
    wrapper: str = "",
    writes: frozenset[str],
) -> None:
    writer = _qrom_calls(function, definitions)
    from itertools import product

    for values in product(*(range(1 << width) for width in selectors.values())):
        entry = {**initial, **dict(zip(selectors, values, strict=True))}
        row = _Row(
            entry, {**selectors, **{label: 3 for label in writes if label == "rot"}}
        )
        row.run(function.statements, qrom=writer, wrapper=wrapper)
        _require(
            not row.locals
            and row.words.keys() == entry.keys()
            and all(
                row.words[key] == value
                for key, value in entry.items()
                if key not in writes
            ),
            function.name,
            "expanded QROM changes its input or leaves dirty work",
        )
