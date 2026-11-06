"""Prove a closed zero-initialized walk cannot leave its input eigenspace."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

from .basis_regions import _name
from .quantum_callbacks import Lambda
from .syntax import Expression, Function, Parameter, Record, Statement, Token, reject
from .unary_selection_proof import _reachable


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


@dataclass(frozen=True)
class _RecordLayout:
    name: str
    fields: Mapping[str, int]

    def zero(self) -> dict[str, int]:
        return {f"{self.name}.{field}": 0 for field in self.fields}

    def field(self, node: Expression) -> str:
        label = _name(node)
        prefix = self.name + "."
        _require(
            label.startswith(prefix) and label[len(prefix) :] in self.fields,
            node.token,
            "operation requires a field of the walk record",
        )
        return label[len(prefix) :]


def _layout(port: Parameter, records: Mapping[str, Record]) -> _RecordLayout:
    record = records.get(port.record or "")
    _require(record is not None, port.name, "walk expects a fixed record")
    assert record is not None
    widths = {}
    for field in record.fields:
        _require(
            field.width is not None and field.record is None and field.length is None,
            field.name,
            "walk records require flat fixed fields",
        )
        widths[field.name.text] = field.width or 0
    return _RecordLayout(port.name.text, widths)


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


def _body(function: Function) -> tuple[Statement, ...]:
    return tuple(step for step in function.statements if step.declaration is None)


def _function(
    definitions: Mapping[str, Function], step: Statement, arity: int
) -> Function:
    function = definitions.get(step.token.text)
    _require(
        _plain(step)
        and function is not None
        and len(step.arguments) == arity
        and len(function.parameters) == arity,
        step.token,
        "proof requires one closed helper with matching arity",
    )
    assert function is not None
    return function


def _one(body: tuple[Statement, ...] | None, token: Token) -> Statement:
    _require(
        body is not None and len(body) == 1, token, "expected one checked operation"
    )
    assert body is not None
    return body[0]


def _labels(arguments: tuple[Expression, ...]) -> tuple[str, ...]:
    return tuple(_name(arg) for arg in arguments)


def _gate(step: Statement, name: str, args: tuple[str, ...]) -> None:
    _require(
        _plain(step) and step.token.text == name and _labels(step.arguments) == args,
        step.token,
        f"expected checked {name} effect on its bound ports",
    )


def _record_prefix(layout: _RecordLayout, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{layout.name}.{field}" for field in fields)


def _check_reflection(
    definitions: Mapping[str, Function], reflection: Function
) -> tuple[Function, str]:
    ctrl, record = reflection.parameters
    _require(
        ctrl.kind == "qbit" and record.record is not None and ctrl.name != record.name,
        reflection.name,
        "reflection needs a distinct scalar control and fixed record",
    )
    scope = _one(_body(reflection), reflection.name)
    inverse = _one(scope.within, scope.token)
    prep_call = _one(inverse.inverse_body, inverse.token)
    prep = _function(definitions, prep_call, 1)
    _require(
        _labels(prep_call.arguments) == (record.name.text,),
        prep_call.token,
        "reflection must prepare its own record",
    )
    projection = _one(scope.apply, scope.token)
    _require(
        projection.within is not None and len(projection.within) == 2,
        projection.token,
        "reflection must complement the entire projected record",
    )
    assert projection.within is not None
    _gate(projection.within[0], "X", (ctrl.name.text,))
    _gate(projection.within[1], "apply_to_all", ("X", record.name.text))
    cast = _one(projection.apply, projection.token)
    wrapper = _function(definitions, cast, 2)
    _check_array_wrapper(wrapper)
    target, callback = cast.arguments
    _require(
        _atom(target, record.name.text)
        and isinstance(callback, Lambda)
        and len(callback.parameters) == 1,
        cast.token,
        "projection must cover the whole record",
    )
    assert isinstance(callback, Lambda)
    condition = _one(callback.body, cast.token)
    _require(
        condition.control_body is not None
        and condition.control_else is None
        and len(condition.arguments) == 1
        and _atom(condition.arguments[0], callback.parameters[0].text),
        condition.token,
        "reflection projector must test the entire zero word",
    )
    _gate(_one(condition.control_body, condition.token), "Z", (ctrl.name.text,))
    return prep, wrapper.name.text


def _phase_control(function: Function, width: int) -> None:
    labels = tuple(p.name.text for p in function.parameters)
    _require(
        len(labels) == width
        and len(set(labels)) == width
        and all(p.kind == "qbit" for p in function.parameters)
        and len(function.statements) == 2,
        function.name,
        "selection phase must use distinct scalar ports",
    )
    first, second = function.statements
    if width == 2:
        _gate(first, "S", (labels[0],))
        _gate(second, "CZ", labels)
        return
    _require(
        len(first.arguments) == len(second.arguments) == 1
        and _atom(first.arguments[0], labels[0])
        and second.arguments[0].token.text == "["
        and _labels(second.arguments[0].children) == labels[:2]
        and first.control_else is None
        and second.control_else is None,
        function.name,
        "selection must condition every phase on the same zero control",
    )
    _gate(_one(first.control_body, first.token), "S", (labels[1],))
    _gate(_one(second.control_body, second.token), "Z", (labels[2],))


def _check_selection(
    definitions: Mapping[str, Function],
    selection: Function,
    records: Mapping[str, Record],
) -> str:
    ell, psi = selection.parameters
    layout = _layout(ell, records)
    _require(
        len(_body(selection)) == 2,
        selection.name,
        "selection requires two pure conjugations",
    )
    first, second = _body(selection)
    _require(
        all(
            scope.within is not None and len(scope.within) == 4
            for scope in (first, second)
        ),
        selection.name,
        "selection must conjugate its phase with closed compute",
    )
    phase1 = _one(first.apply, first.token)
    phase2 = _one(second.apply, second.token)
    inner = _one(phase2.within, phase2.token)
    second_phase = _one(phase2.apply, phase2.token)
    phase_function = _function(definitions, phase1, 2)
    phase_function2 = _function(definitions, second_phase, 3)
    _phase_control(phase_function, 2)
    _phase_control(phase_function2, 3)
    witness = layout.field(phase1.arguments[0])
    _require(
        witness == layout.field(second_phase.arguments[0])
        and layout.fields[witness] == 1,
        selection.name,
        "selection must be identity when its scalar guard is zero",
    )
    flipped = layout.field(second_phase.arguments[1])
    _gate(inner, "X", (f"{layout.name}.{flipped}",))
    _require(flipped != witness, inner.token, "selection cannot modify the zero guard")
    for scope in (first, second):
        assert scope.within is not None
        init, qrom, swap, basis = scope.within
        _require(
            init.token.text == "allocate" and len(init.arguments) == 1 and _plain(init),
            init.token,
            "selection must allocate its private QROM word",
        )
        _require(
            _name(qrom.arguments[-1]) == _name(init.arguments[0])
            and all(layout.field(arg) != witness for arg in qrom.arguments[:-1]),
            qrom.token,
            "selection QROM may only compute from non-guard fields",
        )
        # The two remaining scopes may only borrow spin storage and the
        # private QROM word. Any use of the whole ell record is refused.
        for child in _actions((swap, basis)):
            for argument in child.arguments:
                for symbol in _symbols(argument):
                    _require(
                        not symbol.startswith(ell.name.text + ".")
                        and symbol != ell.name.text,
                        argument.token,
                        "selection conjugation may not touch its record guard",
                    )
    return witness


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


def _check_prepare(
    definitions: Mapping[str, Function],
    prepare: Function,
    records: Mapping[str, Record],
    guard_field: str,
    wrapper: str,
) -> None:
    [port] = prepare.parameters
    layout = _layout(port, records)
    _require(
        len(_body(prepare)) == 2,
        prepare.name,
        "reflection preparation needs two closed stages",
    )
    before_step, after_step = _body(prepare)
    before = _function(definitions, before_step, 1)
    after = _function(definitions, after_step, 1)
    _require(
        all(
            _labels(step.arguments) == (layout.name,)
            for step in (before_step, after_step)
        ),
        prepare.name,
        "reflection preparation stages must borrow its own record",
    )
    after_layout = _layout(after.parameters[0], records)
    _require(
        len(_body(after)) == 6,
        after.name,
        "reflection needs checked address, QROM, and conditional swaps",
    )
    address, sigma, qrom, comparison, auxiliary_h, swap = _body(after)
    address_helper = _function(definitions, address, 3)
    qrom_helper = _function(definitions, qrom, 1)
    _require(
        _labels(qrom.arguments) == (after_layout.name,),
        qrom.token,
        "reflection QROM must borrow its record",
    )
    nu, mu, address_field = tuple(after_layout.field(arg) for arg in address.arguments)
    sigma_field = after_layout.field(sigma.arguments[1])
    aux_field = after_layout.field(auxiliary_h.arguments[0])
    _gate(sigma, "apply_to_all", ("H", f"{after_layout.name}.{sigma_field}"))
    _gate(auxiliary_h, "H", (f"{after_layout.name}.{aux_field}",))
    last_gate = _one(swap.within, swap.token)
    _require(
        last_gate.token.text == "X" and len(last_gate.arguments) == 1,
        swap.token,
        "last preparation permutation needs a checked complemented flag",
    )
    mode_field = after_layout.field(last_gate.arguments[0])
    swap_control = _one(swap.apply, swap.token)
    _require(
        len(swap_control.arguments) == 1
        and swap_control.arguments[0].token.text == "["
        and _labels(swap_control.arguments[0].children)
        == _record_prefix(after_layout, (aux_field, mode_field))
        and swap_control.control_else is None,
        swap_control.token,
        "the last preparation swap must be disabled on the all-zero record",
    )
    swap_gate = _one(swap_control.control_body, swap_control.token)
    swap_helper = _function(definitions, swap_gate, 2)
    _require(
        _labels(swap_gate.arguments) == _record_prefix(after_layout, (mu, nu)),
        swap_gate.token,
        "last preparation swap must exchange the two address fields",
    )
    _check_swap(swap_helper)

    kept = _one(comparison.within, comparison.token)
    _require(
        _plain(kept)
        and kept.token.text == "="
        and len(kept.arguments) == 2
        and kept.arguments[1].token.text == "<"
        and len(kept.arguments[1].children) == 2,
        kept.token,
        "alias-sampling swap must use a strict unsigned comparison",
    )
    keep_field, sample_field = (
        after_layout.field(arg) for arg in kept.arguments[1].children
    )
    _require(
        sample_field == sigma_field
        and keep_field not in {mu, nu, sigma_field, guard_field},
        kept.token,
        "alias comparison must be false on the all-zero record",
    )
    flag = _name(kept.arguments[0])
    for step in comparison.apply:
        if step.control_body is not None:
            _require(
                step.control_else is None
                and len(step.arguments) == 1
                and _atom(step.arguments[0], flag),
                step.token,
                "all alias swaps must use the zero comparison",
            )
            child = _one(step.control_body, step.token)
            _require(
                child.token.text == swap_helper.name.text
                and _plain(child)
                and len(child.arguments) == 2,
                child.token,
                "alias guard permits only a checked swap",
            )
            fields = tuple(after_layout.field(arg) for arg in child.arguments)
            _require(
                guard_field not in fields,
                child.token,
                "alias swaps cannot write the guard",
            )
        else:
            for child in _actions((step,)):
                _require(
                    child.token.text in {"within", "X", "CZ"},
                    child.token,
                    "alias block admits only conditional phases and checked swaps",
                )
                if child.token.text == "X":
                    _gate(child, "X", (flag,))

    # After undoing the last swap, the auxiliary H, and alias sampling, the
    # addresses and guard remain zero. QROM changes only other fields.
    qrom_layout = _layout(qrom_helper.parameters[0], records)
    address_label = f"{qrom_layout.name}.{address_field}"
    protected = {mu, nu, guard_field, mode_field, sigma_field, aux_field, address_field}
    _check_qrom_body(
        qrom_helper,
        qrom_layout.zero(),
        {address_label: qrom_layout.fields[address_field]},
        definitions,
        wrapper=wrapper,
        writes=frozenset(
            f"{qrom_layout.name}.{field}"
            for field in qrom_layout.fields
            if field not in protected
        ),
    )
    _check_address(address_helper, definitions)

    early_layout = _layout(before.parameters[0], records)
    earlier = _body(before)
    _require(
        bool(earlier), before.name, "zero-support preparation must end in a marker"
    )
    mark = earlier[-1]
    for step in _actions(earlier[:-1]):
        for argument in step.arguments:
            forbidden = {early_layout.name, f"{early_layout.name}.{guard_field}"}
            _require(
                not (_symbols(argument) & forbidden),
                argument.token,
                "earlier preparation must preserve the marker bit",
            )
    test = _one(mark.apply, mark.token)
    _require(
        len(test.arguments) == 1
        and test.arguments[0].token.text == "["
        and test.control_else is None,
        test.token,
        "preparation marker needs checked work predicates",
    )
    _gate(
        _one(test.control_body, test.token),
        "X",
        (f"{early_layout.name}.{guard_field}",),
    )
    _require(
        mark.within is not None and len(mark.within) == 2,
        mark.token,
        "preparation marker must compute its predicate and one reversible update",
    )
    assert mark.within is not None
    make, update = mark.within
    predicate = _function(definitions, make, len(make.arguments))
    predicate_ports = predicate.parameters
    _require(
        len(predicate_ports) >= 3
        and all(p.role == "output" for p in predicate_ports[2:])
        and _labels(make.arguments[:2]) == _record_prefix(early_layout, (mu, nu)),
        make.token,
        "marker predicates must read only the two zero address fields",
    )
    actual_names = _labels(make.arguments[2:])
    known = _Row({**early_layout.zero(), **dict.fromkeys(actual_names, 0)})
    first_two = {p.name.text for p in predicate_ports[:2]}
    assigned = set()
    for statement in predicate.statements:
        _require(
            _plain(statement)
            and statement.token.text == "="
            and len(statement.arguments) == 2,
            statement.token,
            "marker must assign each fresh Boolean output",
        )
        destination, value = statement.arguments
        label = _name(destination)
        _require(
            label not in assigned,
            statement.token,
            "marker predicate reads an unknown value",
        )
        formal_names = [p.name.text for p in predicate_ports[2:]]
        _require(
            label in formal_names,
            destination.token,
            "predicate may only write fresh outputs",
        )
        known.words[actual_names[formal_names.index(label)]] = _zero_predicate(
            value, first_two
        )
        assigned.add(label)
    _require(
        assigned == {p.name.text for p in predicate_ports[2:]},
        make.token,
        "marker predicates did not initialize all declared outputs",
    )
    known.run((update,))
    _check_early_preparation(
        earlier[:-1], make, update, test.arguments[0], early_layout, mu, nu, mode_field
    )
    _require(
        all(known.read(arg) == 1 for arg in test.arguments[0].children),
        test.token,
        "prepared zero word is not orthogonal to the marked subspace",
    )


def _expression_key(node: Expression) -> tuple[str, tuple[object, ...]]:
    return node.token.text, tuple(_expression_key(child) for child in node.children)


def _same_call(first: Statement, second: Statement) -> bool:
    return (
        _plain(first)
        and _plain(second)
        and first.token.text == second.token.text
        and tuple(map(_expression_key, first.arguments))
        == tuple(map(_expression_key, second.arguments))
    )


def _check_early_preparation(
    body: tuple[Statement, ...],
    predicates: Statement,
    update: Statement,
    flag_list: Expression,
    layout: _RecordLayout,
    mu: str,
    nu: str,
    mode: str,
) -> None:
    _require(len(body) == 5, predicates.token, "unexpected coherent preparation stage")
    first, second, marked, diffuser, mode_xor = body
    mu_label, nu_label, mode_label = _record_prefix(layout, (mu, nu, mode))
    _gate(first, "apply_to_all", ("H", mu_label))
    _gate(second, "apply_to_all", ("H", nu_label))
    _require(
        marked.within is not None and len(marked.within) == 4,
        marked.token,
        "initial phase needs a closed marker compute",
    )
    assert marked.within is not None
    rotation, inner_mode, inner_predicates, inner_update = marked.within
    _require(
        _plain(rotation)
        and rotation.token.text == "RY"
        and len(rotation.arguments) == 2,
        rotation.token,
        "marker rotation must be a static scalar gate",
    )
    import math

    angle = rotation.arguments[0]
    try:
        numeric = float(angle.token.text)
    except ValueError:
        reject(
            angle.token, "zero_walk", "marker rotation must have a finite literal angle"
        )
    _require(
        not angle.children and math.isfinite(numeric),
        angle.token,
        "marker rotation must be finite",
    )
    branch = layout.field(rotation.arguments[1])
    _require(
        layout.fields[branch] == 1 and branch != mode,
        rotation.token,
        "marker rotation needs its own scalar",
    )
    r = f"{layout.name}.{branch}"
    _require(
        _same_call(inner_mode, mode_xor)
        and _same_call(inner_predicates, predicates)
        and _same_call(inner_update, update),
        marked.token,
        "coherent marker must use the same clean predicates",
    )
    _require(
        mode_xor.token.text == "^="
        and len(mode_xor.arguments) == 2
        and _name(mode_xor.arguments[0]) == mode_label
        and mode_xor.arguments[1].token.text in {"==", "!=", "<", ">", "<=", ">="},
        mode_xor.token,
        "mode update must be a checked XOR comparison",
    )
    chosen = _one(marked.apply, marked.token)
    _require(
        len(chosen.arguments) == 1
        and chosen.control_else is None
        and _expression_key(chosen.arguments[0]) == _expression_key(flag_list),
        chosen.token,
        "initial phase must use the checked predicates",
    )
    _gate(_one(chosen.control_body, chosen.token), "Z", (r,))
    _require(
        diffuser.within is not None and len(diffuser.within) == 2,
        diffuser.token,
        "diffuser must conjugate a static phase",
    )
    assert diffuser.within is not None
    _gate(diffuser.within[0], "apply_to_all", ("H", mu_label))
    _gate(diffuser.within[1], "apply_to_all", ("H", nu_label))
    condition = _one(diffuser.apply, diffuser.token)
    _require(
        len(condition.arguments) == 1
        and condition.arguments[0].token.text == "["
        and _labels(condition.arguments[0].children) == (mu_label, nu_label)
        and condition.control_else is None,
        condition.token,
        "diffuser phase must read the two address words",
    )
    _gate(_one(condition.control_body, condition.token), "Z", (r,))


def _zero_predicate(node: Expression, zero_names: set[str]) -> int:
    from fractions import Fraction

    def number(arg: Expression) -> Fraction:
        if arg.children:
            reject(
                arg.token, "zero_walk", "marker expression must use scalar comparisons"
            )
        if arg.token.text in zero_names:
            return Fraction(0)
        try:
            return Fraction(arg.token.text)
        except ValueError:
            reject(
                arg.token,
                "zero_walk",
                "marker expression has an unknown classical value",
            )

    if not node.children:
        value = number(node)
        _require(value in {0, 1}, node.token, "marker literal must be Boolean")
        return int(value)
    _require(
        len(node.children) == 2, node.token, "marker comparison must have two operands"
    )
    a, b = (number(arg) for arg in node.children)
    comparisons = {
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
        "==": a == b,
        "!=": a != b,
    }
    _require(node.token.text in comparisons, node.token, "unsupported marker predicate")
    return int(comparisons[node.token.text])


def _check_swap(function: Function) -> None:
    loop = _one(function.statements, function.name)
    ports = tuple(port.name.text for port in function.parameters)
    _require(
        len(ports) == 2
        and loop.repeat_index is not None
        and len(loop.arguments) == 1
        and _name(loop.arguments[0]) == ports[0] + ".len",
        loop.token,
        "swap helper must iterate the two equal-width arrays",
    )
    gate = _one(loop.repeat_body, loop.token)
    assert loop.repeat_index is not None
    _require(
        _plain(gate)
        and gate.token.text == "SWAP"
        and len(gate.arguments) == 2
        and all(
            arg.token.text == "["
            and _labels(arg.children) == (port, loop.repeat_index.text)
            for arg, port in zip(gate.arguments, ports, strict=True)
        ),
        gate.token,
        "swap helper must exchange corresponding bits",
    )


def _check_address(function: Function, definitions: Mapping[str, Function]) -> None:
    nu, mu, address = (port.name.text for port in function.parameters)
    scope = _one(_body(function), function.name)
    prepare = _one(scope.within, scope.token)
    shift = _function(definitions, prepare, 2)
    _require(
        _atom(prepare.arguments[0], nu),
        prepare.token,
        "address helper must only compute a private split",
    )
    temporary = _name(prepare.arguments[1])
    update = _one(scope.apply, scope.token)
    _require(
        _plain(update)
        and update.token.text == "^="
        and len(update.arguments) == 2
        and _atom(update.arguments[0], address)
        and _symbols(update.arguments[1]) <= {nu, mu, temporary, "1"},
        update.token,
        "address arithmetic may only XOR its own field",
    )
    bind_scope = _one(_body(shift), shift.name)
    bind = _one(bind_scope.within, bind_scope.token)
    assign = _one(bind_scope.apply, bind_scope.token)
    ports = tuple(port.name.text for port in shift.parameters)
    _require(
        _plain(bind)
        and bind.token.text == "->"
        and len(bind.arguments) == 2
        and _atom(bind.arguments[0], ports[0])
        and bind.arguments[1].token.text == "{"
        and len(bind.arguments[1].children) == 2,
        bind.token,
        "address split must borrow one quantum input and restore it",
    )
    _gate(assign, "=", (ports[1], _name(bind.arguments[1].children[0])))


def _check_basis(function: Function, definitions: Mapping[str, Function]) -> None:
    ports = function.parameters
    _require(
        len(ports) == 2 and ports[0].kind == "qnum[]" and ports[1].kind == "qbit[]",
        function.name,
        "basis change needs a numeric rotation array and a spin view",
    )
    loop = _one(function.statements, function.name)
    _require(
        loop.repeat_index is not None
        and len(loop.arguments) == 1
        and len(loop.repeat_body) == 2,
        loop.token,
        "basis change must iterate paired rotations",
    )
    assert loop.repeat_index is not None
    bound = loop.arguments[0]
    _require(
        bound.token.text == "-"
        and _labels(bound.children) == (ports[0].name.text + ".len", "1"),
        loop.token,
        "basis change bound must match the rotation array",
    )
    rotation: Function | None = None
    for scope in loop.repeat_body:
        _require(
            scope.within is not None
            and len(scope.within) == 2
            and len(scope.apply) == 1,
            scope.token,
            "basis change uses two checked local conjugations",
        )
        assert scope.within is not None
        for call, gates in zip(scope.within, (("H",), ("SDG", "H")), strict=True):
            basis = _function(definitions, call, 1)
            _require(
                call.arguments[0].token.text == "["
                and len(call.arguments[0].children) == 2
                and _atom(call.arguments[0].children[0], ports[1].name.text),
                call.token,
                "basis gate must target only the spin view",
            )
            _require(
                len(basis.statements) == len(gates),
                basis.name,
                "basis transform is outside the checked gate set",
            )
            for step, name in zip(basis.statements, gates, strict=True):
                _gate(step, name, (basis.parameters[0].name.text,))
        call = scope.apply[0]
        if call.inverse_body is not None:
            call = _one(call.inverse_body, call.token)
        bound_rotation = _function(definitions, call, 2)
        _require(
            rotation is None or bound_rotation is rotation,
            call.token,
            "basis change must use a common digital rotation",
        )
        rotation = bound_rotation
        _require(
            call.arguments[0].token.text == "["
            and _labels(call.arguments[0].children)
            == (ports[0].name.text, loop.repeat_index.text)
            and call.arguments[1].token.text == "{"
            and len(call.arguments[1].children) == 2,
            call.token,
            "digital rotation needs two checked spin positions",
        )
    assert rotation is not None
    turn = _one(rotation.statements, rotation.name)
    _require(
        turn.repeat_index is not None
        and len(turn.arguments) == 1
        and _name(turn.arguments[0]) == rotation.parameters[0].name.text + ".len",
        turn.token,
        "digital rotation must iterate over its fixed bits",
    )
    leaf = _one(turn.repeat_body, turn.token)
    phase = _function(definitions, leaf, 3)
    _require(
        _atom(leaf.arguments[2], rotation.parameters[1].name.text)
        and leaf.arguments[1].token.text == "["
        and turn.repeat_index is not None
        and _labels(leaf.arguments[1].children)
        == (rotation.parameters[0].name.text, turn.repeat_index.text),
        leaf.token,
        "digital rotation may only use its fixed read-only bits",
    )
    scope = _one(phase.statements, phase.name)
    parity = _one(scope.within, scope.token)
    effect = _one(scope.apply, scope.token)
    _require(
        _plain(parity)
        and parity.token.text == "CX"
        and _plain(effect)
        and effect.token.text == "CRZ"
        and len(parity.arguments) == 2
        and len(effect.arguments) == 3,
        scope.token,
        "digital two-spin rotation requires a checked parity conjugation",
    )


def _check_spin_conjugations(
    select: Function,
    definitions: Mapping[str, Function],
    records: Mapping[str, Record],
) -> None:
    ell = _layout(select.parameters[0], records)
    psi = _layout(select.parameters[1], records)
    for scope in _body(select):
        assert scope.within is not None
        allocation, read, swap, basis = scope.within
        slot = _name(allocation.arguments[0])
        matching = [
            step.declaration
            for step in select.statements
            if step.declaration is not None and step.declaration.name.text == slot
        ]
        _require(
            len(matching) == 1
            and matching[0].kind == "qnum[]"
            and matching[0].width == 4
            and matching[0].length == 3,
            allocation.token,
            "digital QROM needs a checked three-element four-bit array",
        )
        reader = _function(definitions, read, len(read.arguments))
        selectors = {
            formal.name.text: ell.fields[ell.field(actual)]
            for formal, actual in zip(
                reader.parameters[:-1], read.arguments[:-1], strict=True
            )
        }
        _require(
            all(1 <= bits <= 8 for bits in selectors.values())
            and sum(selectors.values()) <= 8,
            reader.name,
            "expanded digital QROM exceeds the exhaustive proof bound",
        )
        target = reader.parameters[-1].name.text
        _check_qrom_body(
            reader,
            {target: 0},
            selectors,
            definitions,
            writes=frozenset({target}),
        )
        _require(
            swap.control_else is None
            and len(swap.arguments) == 1
            and psi.fields[psi.field(swap.arguments[0])] == 1,
            swap.token,
            "spin exchange needs a scalar control on its own record",
        )
        exchange = _one(swap.control_body, swap.token)
        helper = _function(definitions, exchange, 2)
        _check_swap(helper)
        a, b = (psi.field(arg) for arg in exchange.arguments)
        _require(
            a != b and psi.fields[a] == psi.fields[b] == 3,
            exchange.token,
            "spin exchange must borrow two separate equal-width spin views",
        )
        basis_call = _one(basis.inverse_body, basis.token)
        transform = _function(definitions, basis_call, 2)
        _require(
            _atom(basis_call.arguments[0], slot)
            and psi.field(basis_call.arguments[1]) in {a, b},
            basis_call.token,
            "basis transform can only use the private rotation array and spin view",
        )
        _check_basis(transform, definitions)


@dataclass(frozen=True)
class ZeroWalk:
    definitions: Mapping[str, Function]
    record_width: int
    verified_control_words: int


def prove_zero_walk(
    definitions: Mapping[str, Function], records: Mapping[str, Record]
) -> ZeroWalk:
    main = definitions["main"]
    calls = {
        name: {step.token.text for step in _actions(function.statements)}
        for name, function in definitions.items()
    }
    _require(
        set(definitions) == _reachable("main", calls),
        main.name,
        "zero walk requires a closed reachable call graph",
    )
    _require(
        all(
            f.disable_const_checks is None and f.disable_permutation_check is None
            for f in definitions.values()
        ),
        main.name,
        "zero walk requires enabled source checks",
    )
    ports = {port.name.text: port for port in main.parameters}
    body = main.statements
    _require(
        len(ports) == len(main.parameters) == 3
        and all(p.role == "output" for p in ports.values())
        and len(body) == 7
        and not main.permutation,
        main.name,
        "zero walk needs three freshly allocated outputs and a static tail",
    )
    allocations, launch, prepare_spin, execute, finish = body[:3], *body[3:]
    _require(
        all(
            step.token.text == "allocate" and _plain(step) and len(step.arguments) == 1
            for step in allocations
        )
        and {_name(step.arguments[0]) for step in allocations} == set(ports),
        main.name,
        "zero-walk outputs must all start from zero",
    )
    clock, record, spin = _labels(execute.arguments)
    _require(
        set((clock, record, spin)) == set(ports),
        execute.token,
        "walk must borrow the three distinct declared outputs",
    )
    t, ell, psi = (ports[name] for name in (clock, record, spin))
    _require(
        t.kind == "qbit[]"
        and t.width is not None
        and 1 <= t.width <= 6
        and ell.record is not None
        and psi.record is not None,
        main.name,
        "zero walk needs a bounded phase counter and two records",
    )
    assert t.width is not None
    launcher = _function(definitions, launch, 1)
    initializer = _function(definitions, prepare_spin, 1)
    _require(
        _labels(launch.arguments) == (clock,)
        and _labels(prepare_spin.arguments) == (spin,),
        main.name,
        "walk record must remain zero during public state preparation",
    )
    _require(
        launcher.parameters[0].kind == "qbit[]"
        and len(launcher.statements) == 1
        and launcher.statements[0].token.text == "inplace_prepare_amplitudes"
        and len(launcher.statements[0].arguments) == 3
        and _atom(
            launcher.statements[0].arguments[-1], launcher.parameters[0].name.text
        ),
        launcher.name,
        "phase counter requires a closed amplitude preparation",
    )
    spin_layout = _layout(initializer.parameters[0], records)
    for step in initializer.statements:
        _require(
            len(step.arguments) == 1
            and spin_layout.fields[spin_layout.field(step.arguments[0])] == 1,
            step.token,
            "spin initialization admits scalar Hadamards",
        )
        _gate(step, "H", (_name(step.arguments[0]),))
    _gate(_one(finish.inverse_body, finish.token), "qft", (clock,))
    walk = _function(definitions, execute, 3)
    w_t, w_ell, w_psi = walk.parameters
    _require(
        w_t.kind == t.kind
        and w_ell.record == ell.record
        and w_psi.record == psi.record,
        walk.name,
        "walk call types disagree with its output layouts",
    )
    invoked: set[str] = {
        step.token.text
        for step in _actions(walk.statements)
        if step.token.text in definitions
    }
    _require(
        len(invoked) == 2,
        walk.name,
        "walk may call one select and one reflection helper",
    )
    _require(
        all(len(definitions[name].parameters) == 2 for name in invoked),
        walk.name,
        "walk helper signatures are outside the proved profile",
    )
    reflections = [
        definitions[name]
        for name in invoked
        if definitions[name].parameters[0].kind == "qbit"
    ]
    _require(
        len(reflections) == 1, walk.name, "walk needs one scalar-controlled reflection"
    )
    [reflection] = reflections
    [selection] = [
        definitions[name] for name in invoked if name != reflection.name.text
    ]
    _require(
        reflection.parameters[1].record == ell.record
        and selection.parameters[0].record == ell.record
        and selection.parameters[1].record == psi.record,
        walk.name,
        "select or reflection must use the same borrowed record types",
    )
    prep, wrapper = _check_reflection(definitions, reflection)
    _require(
        prep.parameters[0].record == ell.record,
        prep.name,
        "reflection preparation type differs",
    )
    zero = _check_selection(definitions, selection, records)
    _check_spin_conjugations(selection, definitions, records)
    _check_prepare(definitions, prep, records, zero, wrapper)

    t_name = w_t.name.text
    for word in range(1 << t.width):
        row = _Row({t_name: word}, {t_name: t.width})
        row.run(
            walk.statements,
            allowed={
                reflection.name.text: ("$control", w_ell.name.text),
                selection.name.text: (w_ell.name.text, w_psi.name.text),
            },
        )
        _require(
            row.words == {t_name: word} and not row.locals,
            walk.name,
            "walk control must return the phase counter and all its work clean",
        )
    reduced = replace(main, statements=(*body[:5], body[-1]))
    return ZeroWalk(
        {
            "main": reduced,
            launcher.name.text: launcher,
            initializer.name.text: initializer,
        },
        ell.width or 0,
        1 << t.width,
    )
