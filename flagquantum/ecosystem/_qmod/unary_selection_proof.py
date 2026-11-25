"""Prove an annotated reversible selector by bounded basis evaluation."""

from dataclasses import dataclass, replace
from typing import cast

from .integers import checked_integer, integer_operation
from .syntax import Expression, Function, Parameter, Statement, Token, reject

_COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


@dataclass
class _Bit:
    value: int | None


_Actual = int | tuple[_Bit, ...]


def _plain(statement: Statement) -> bool:
    return statement == Statement(statement.token, statement.arguments)


class _Proof:
    def __init__(self, definitions: dict[str, Function], root: Function) -> None:
        self.definitions = definitions
        self.root = root
        self.selector, self.data = root.parameters
        self.steps = 0
        self.calls: list[Statement] = []
        self.visited: set[str] = set()
        self.leaves: set[str] = set()

    def expression(self, node: Expression, bound: dict[str, _Actual]) -> _Actual:
        text = node.token.text
        if not node.children:
            if text.isdecimal():
                return checked_integer(int(text), node.token)
            if text in bound:
                return bound[text]
        if text in {"+", "-", "*"} and len(node.children) == 2:
            a, b = (self.integer(child, bound) for child in node.children)
            return integer_operation(node.token, a, b)
        if text in _COMPARISONS and len(node.children) == 2:
            a, b = (self.integer(child, bound) for child in node.children)
            return int(_COMPARISONS[text](a, b))
        if text in {"[", ":"} and len(node.children) in {2, 3}:
            source = self.word(node.children[0], bound)
            start = self.integer(node.children[1], bound)
            stop = start + 1 if text == "[" else self.integer(node.children[2], bound)
            if not 0 <= start < stop <= len(source):
                reject(node.token, "width", "selector view is out of bounds")
            return source[start:stop]
        reject(
            node.token, "selector", "selector proof needs a bounded static expression"
        )

    def integer(self, node: Expression, bound: dict[str, _Actual]) -> int:
        value = self.expression(node, bound)
        if isinstance(value, int):
            return value
        if len(value) == 1 and value[0].value is not None:
            return value[0].value
        reject(node.token, "selector", "selector proof cannot branch on data")

    def word(self, node: Expression, bound: dict[str, _Actual]) -> tuple[_Bit, ...]:
        value = self.expression(node, bound)
        if isinstance(value, int):
            reject(node.token, "selector", "selector proof needs a quantum view")
        return value

    def bind(
        self, function: Function, actual: tuple[_Actual, ...], active: tuple[str, ...]
    ) -> None:
        name = function.name.text
        if name in active or len(active) >= 24 or function.permutation:
            reject(
                function.name,
                "selector",
                "selector helper graph must be bounded and acyclic",
            )
        self.visited.add(name)
        if len(actual) != len(function.parameters) or len(
            {p.name.text for p in function.parameters}
        ) != len(actual):
            reject(
                function.name, "selector", "selector helper argument count disagrees"
            )
        bound: dict[str, _Actual] = {}
        for port, value in zip(function.parameters, actual, strict=True):
            if (
                port.role == "integer"
                and port.kind == "int"
                and isinstance(value, int)
                or (
                    port.role in {"const", "inout"}
                    and not isinstance(value, int)
                    and port.kind in {"qbit", "qnum", "qbit[]"}
                    and port.record is None
                    and (port.width is None or len(value) == port.width)
                    and all(
                        (bit.value is None) == (port.role == "inout") for bit in value
                    )
                )
            ):
                bound[port.name.text] = value
            else:
                reject(
                    port.name,
                    "selector",
                    "selector ports require typed borrowed bits or a static integer",
                )
        if len(actual) == 1 and function.parameters[0].role == "inout":
            data = bound[function.parameters[0].name.text]
            if data is not self.data_bits:
                reject(
                    function.name,
                    "alias",
                    "selector leaf requires the complete data view",
                )
            # The existing adapter verifies and lowers the leaf as a pure unitary.
            self.calls.append(Statement(function.name, (Expression(self.data.name),)))
            self.leaves.add(name)
            return

        writable = function.disable_const_checks
        if writable is not None and not any(
            port.name.text == writable.text
            and port.role == "const"
            and port.kind == "qbit"
            for port in function.parameters
        ):
            reject(
                writable, "ownership", "selector annotation must name a borrowed qbit"
            )
        if len(
            {id(bit) for value in actual if isinstance(value, tuple) for bit in value}
        ) != sum(len(value) for value in actual if isinstance(value, tuple)):
            reject(function.name, "alias", "selector helper arguments must be disjoint")
        original = dict(bound)
        saved = {
            id(bit): bit.value
            for value in bound.values()
            if isinstance(value, tuple)
            for bit in value
        }
        locals_: dict[str, Parameter] = {}
        writable_bits = (
            {id(bit) for bit in cast(tuple[_Bit, ...], bound[writable.text])}
            if writable is not None
            else set()
        )
        self.block(function.statements, bound, locals_, writable_bits, (*active, name))
        for label, value in original.items():
            if bound.get(label) is not value:
                reject(
                    function.name,
                    "ownership",
                    "selector helper did not restore its formal views",
                )
            if isinstance(value, tuple) and any(
                saved[id(bit)] != bit.value for bit in value
            ):
                reject(
                    function.name, "ownership", "selector helper changed a borrowed bit"
                )
        if any(local in bound for local in locals_):
            reject(function.name, "ownership", "selector helper leaked local scratch")

    def block(
        self,
        statements: tuple[Statement, ...],
        bound: dict[str, _Actual],
        locals_: dict[str, Parameter],
        writable: set[int],
        active: tuple[str, ...],
        *,
        inverse: bool = False,
    ) -> None:
        for statement in reversed(statements) if inverse else statements:
            self.steps += 1
            token = statement.token
            if self.steps > 50_000:
                reject(
                    token, "budget.control", "selector proof exceeded its visit budget"
                )
            args = statement.arguments
            if statement.static_if_body is not None:
                if self.integer(args[0], bound):
                    self.block(
                        statement.static_if_body,
                        bound,
                        locals_,
                        writable,
                        active,
                        inverse=inverse,
                    )
                continue
            if statement.control_body is not None:
                chosen = (
                    statement.control_body
                    if self.integer(args[0], bound)
                    else statement.control_else
                )
                if chosen is not None:
                    self.block(
                        chosen, bound, locals_, writable, active, inverse=inverse
                    )
                continue
            if statement.within is not None and not inverse:
                self.block(statement.within, bound, locals_, writable, active)
                self.block(statement.apply, bound, locals_, writable, active)
                self.block(
                    statement.within, bound, locals_, writable, active, inverse=True
                )
                continue
            if statement.declaration is not None and not inverse:
                declaration = statement.declaration
                if (
                    declaration.name.text in locals_
                    or declaration.name.text in bound
                    or declaration.kind not in {"qbit", "qbit[]"}
                ):
                    reject(
                        declaration.name,
                        "ownership",
                        "selector proof needs fresh local bits",
                    )
                locals_[declaration.name.text] = declaration
                continue
            if not _plain(statement):
                reject(token, "selector", "unsupported selector statement")
            name = token.text
            if name in {"X", "CX"} and len(args) == (1 if name == "X" else 2):
                target_word = self.word(args[-1], bound)
                if len(target_word) != 1:
                    reject(
                        token, "width", "selector bookkeeping requires a scalar target"
                    )
                [target] = target_word
                if target.value is None or id(target) not in writable:
                    reject(
                        token,
                        "selector",
                        "selector bookkeeping may only target borrowed flags",
                    )
                if name == "CX":
                    source = self.word(args[0], bound)
                    if len(source) != 1 or source[0] is target:
                        reject(token, "alias", "selector CX needs distinct scalar bits")
                    target.value ^= self.integer(args[0], bound)
                else:
                    target.value ^= 1
                continue
            if inverse:
                reject(
                    token,
                    "selector",
                    "selector preparation must have a reversible bit inverse",
                )
            if name == "allocate" and len(args) == 2:
                local = args[1].token.text
                if (
                    args[1].children
                    or local not in locals_
                    or local in bound
                    or locals_[local].kind != "qbit"
                    or self.integer(args[0], bound) != 1
                ):
                    reject(
                        token,
                        "ownership",
                        "selector scratch requires a fresh local qbit",
                    )
                scratch = _Bit(0)
                bound[local] = (scratch,)
                writable.add(id(scratch))
            elif name == "free" and len(args) == 1:
                local = args[0].token.text
                if (
                    args[0].children
                    or local not in locals_
                    or self.integer(args[0], bound) != 0
                ):
                    reject(
                        token, "ownership", "selector scratch did not return to zero"
                    )
                (scratch,) = self.word(args[0], bound)
                if id(scratch) not in writable:
                    reject(
                        token,
                        "ownership",
                        "free requires locally owned selector scratch",
                    )
                writable.remove(id(scratch))
                bound.pop(local)
            elif name == "->" and len(args) == 2:
                left, right = (arg.token.text for arg in args)
                if (
                    any(arg.children for arg in args)
                    or right in bound
                    or left not in bound
                ):
                    reject(
                        token,
                        "ownership",
                        "selector binding must move one complete view",
                    )
                port = locals_.get(right) or next(
                    (
                        port
                        for port in self.definitions[active[-1]].parameters
                        if port.name.text == right
                    ),
                    None,
                )
                source = self.word(args[0], bound)
                if (
                    port is None
                    or port.kind not in {"qnum", "qbit", "qbit[]"}
                    or any(bit.value is None for bit in source)
                    or port.width is not None
                    and len(source) != port.width
                ):
                    reject(
                        token,
                        "binding",
                        "selector binding requires an exact borrowed bit view",
                    )
                bound[right] = bound.pop(left)
            elif name in self.definitions:
                actual = tuple(self.expression(arg, bound) for arg in args)
                self.bind(self.definitions[name], actual, active)
            else:
                reject(token, "selector", "unsupported selector bookkeeping operation")

    def run(self, width: int, data_width: int) -> tuple[tuple[Statement, ...], ...]:
        result = []
        self.data_bits = tuple(_Bit(None) for _ in range(data_width))
        for word in range(1 << width):
            self.calls = []
            flags = tuple(_Bit(word >> i & 1) for i in range(width))
            self.bind(self.root, (flags, self.data_bits), ())
            result.append(tuple(self.calls))
        return tuple(result)


def _children(statement: Statement) -> tuple[Statement, ...]:
    return (
        *(statement.within or ()),
        *statement.apply,
        *(statement.control_body or ()),
        *(statement.control_else or ()),
        *(statement.static_if_body or ()),
        *statement.repeat_body,
        *(statement.inverse_body or ()),
    )


def _calls(function: Function) -> set[str]:
    pending = list(function.statements)
    result = set()
    while pending:
        statement = pending.pop()
        result.add(statement.token.text)
        pending.extend(_children(statement))
    return result


def _reachable(name: str, calls: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    pending = [name]
    while pending:
        callee = pending.pop()
        if callee in calls and callee not in visited:
            visited.add(callee)
            pending.extend(calls[callee])
    return visited


def prove_annotated_selections(definitions: dict[str, Function]) -> dict[str, Function]:
    """Replace a proved clean selector with existing coherent static cases."""
    annotations = {
        name for name, f in definitions.items() if f.disable_const_checks is not None
    }
    if not annotations:
        return definitions
    result = dict(definitions)
    direct = {name: _calls(function) for name, function in definitions.items()}
    for root in definitions.values():
        if (
            root.name.text not in result
            or len(root.parameters) != 2
            or root.permutation
        ):
            continue
        selector, data = root.parameters
        width, data_width = selector.width, data.width
        if (
            selector.role != "const"
            or selector.kind != "qnum"
            or data.role != "inout"
            or data.kind != "qbit[]"
            or width is None
            or data_width is None
            or not 1 <= width <= 6
            or not 1 <= data_width <= 8
            or not _reachable(root.name.text, direct) & annotations & result.keys()
        ):
            continue
        proof = _Proof(result, root)
        table = proof.run(width, data_width)
        internal = proof.visited - proof.leaves - {root.name.text}
        if any(
            direct[name] & internal
            for name in result
            if name not in internal and name != root.name.text
        ):
            reject(root.name, "ownership", "selector bookkeeping helpers cannot escape")
        if not internal & annotations:
            continue
        for name in internal:
            result.pop(name)
        token = root.name

        def expression(
            text: str, *children: Expression, origin: Token = token
        ) -> Expression:
            return Expression(replace(origin, text=text), children)

        index = replace(token, text="$selector_word")
        index_expr = Expression(index)
        cases = tuple(
            Statement(
                replace(token, text="if"),
                (expression("==", index_expr, expression(str(word))),),
                static_if_body=gates,
            )
            for word, gates in enumerate(table)
            if gates
        )
        condition = expression("==", Expression(selector.name), index_expr)
        control = Statement(
            replace(token, text="control"), (condition,), control_body=cases
        )
        loop = Statement(
            replace(token, text="repeat"),
            (expression(str(1 << width)),),
            repeat_index=index,
            repeat_body=(control,),
        )
        result[root.name.text] = replace(root, statements=(loop,))
    return result
