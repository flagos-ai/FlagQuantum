"""Prove preparation-time consuming permutations over a bounded basis domain."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from .integers import checked_integer, integer_operation
from .syntax import Expression, Function, Parameter, Statement, Token, reject
from .unary_selection_proof import _calls, _reachable


@dataclass(eq=False)
class _Wire:
    bit: int


def _atom(node: Expression) -> str:
    if node.children:
        reject(node.token, "ownership", "consuming bindings require whole ports")
    return node.token.text


class _BasisWiring:
    def __init__(self, definitions: Mapping[str, Function]) -> None:
        self.definitions = definitions
        self.visits = 0

    def invoke(
        self,
        function: Function,
        inputs: dict[str, tuple[_Wire, ...]],
        active: tuple[str, ...] = (),
    ) -> dict[str, tuple[_Wire, ...]]:
        name = function.name.text
        if (
            name in active
            or len(active) >= 24
            or not function.permutation
            or function.disable_const_checks is not None
            or function.disable_permutation_check is not None
        ):
            reject(
                function.name,
                "ownership",
                "consuming preparation needs a closed reversible call graph",
            )
        parameters = {port.name.text: port for port in function.parameters}
        if len(parameters) != len(function.parameters) or any(
            port.kind not in {"qbit", "qbit[]", "qnum"}
            or port.record is not None
            or port.length is not None
            or port.fractional_bits
            or port.width is None
            or not 1 <= port.width <= 8
            or port.role not in {"input", "inout", "output"}
            for port in function.parameters
        ):
            reject(
                function.name,
                "ownership",
                "consuming proof requires distinct fixed unsigned ports",
            )
        for statement in function.statements:
            if statement.declaration is not None:
                port = statement.declaration
                if port.name.text in parameters:
                    reject(port.name, "ownership", "consuming local name must be fresh")
                parameters[port.name.text] = port
        live = dict(inputs)
        self.block(function.statements, parameters, live, {}, (*active, name))
        retained = {
            port.name.text
            for port in function.parameters
            if port.role in {"inout", "output"}
        }
        if live.keys() != retained:
            reject(
                function.name,
                "ownership",
                "consuming permutation must transfer or free every owned wire",
            )
        for port in function.parameters:
            if port.role == "inout" and live[port.name.text] != inputs[port.name.text]:
                reject(
                    port.name,
                    "ownership",
                    "writable borrowed port must be restored in order",
                )
        return {
            name: value
            for name, value in live.items()
            if parameters[name].role == "output"
        }

    def block(
        self,
        body: tuple[Statement, ...],
        parameters: dict[str, Parameter],
        live: dict[str, tuple[_Wire, ...]],
        indices: dict[str, int],
        active: tuple[str, ...],
        controls: frozenset[_Wire] = frozenset(),
    ) -> None:
        def integer(node: Expression) -> int:
            token = node.token
            if not node.children:
                if token.text.isdecimal():
                    return checked_integer(int(token.text), token)
                if token.text in indices:
                    return indices[token.text]
            if token.text in {"+", "-", "*"} and len(node.children) == 2:
                return integer_operation(
                    token, *(integer(child) for child in node.children)
                )
            reject(
                token, "ownership", "consuming permutation needs static integer indices"
            )

        def view(node: Expression) -> tuple[_Wire, ...]:
            if node.token.text == "[":
                source, index = node.children
                bits = view(source)
                offset = integer(index)
                if not 0 <= offset < len(bits):
                    reject(
                        index.token, "width", "consuming view index is out of bounds"
                    )
                return (bits[offset],)
            bound = live.get(_atom(node))
            if bound is None:
                reject(
                    node.token,
                    "ownership",
                    "consumed or uninitialized port cannot be read",
                )
            return bound

        def scalar(node: Expression) -> _Wire:
            bits = view(node)
            if len(bits) != 1:
                reject(node.token, "ownership", "consuming gate needs scalar bits")
            return bits[0]

        def bind(node: Expression, wires: tuple[_Wire, ...]) -> None:
            label = _atom(node)
            port = parameters.get(label)
            if (
                port is None
                or port.role not in {"output", "local", "inout"}
                or label in live
                or port.width != len(wires)
            ):
                reject(
                    node.token,
                    "ownership",
                    "consuming destination must be fresh with matching width",
                )
            live[label] = wires

        for statement in body:
            self.visits += 1
            token = statement.token
            if self.visits > 30_000:
                reject(
                    token,
                    "budget.control",
                    "consuming permutation exceeds its proof budget",
                )
            args = statement.arguments
            op = token.text
            if statement.declaration is not None:
                continue
            if statement.repeat_index is not None:
                count = integer(args[0])
                index = statement.repeat_index
                if not 0 <= count <= 16 or index.text in indices:
                    reject(
                        index,
                        "budget.control",
                        "consuming loop needs a fresh bounded index",
                    )
                for step in range(count):
                    self.block(
                        statement.repeat_body,
                        parameters,
                        live,
                        {**indices, index.text: step},
                        active,
                        controls,
                    )
                continue
            if statement.control_body is not None and len(args) == 1:
                control = scalar(args[0])
                branch = (
                    statement.control_body if control.bit else statement.control_else
                )
                if branch is not None:
                    self.block(
                        branch, parameters, live, indices, active, controls | {control}
                    )
                continue
            if statement != Statement(token, args):
                reject(token, "ownership", "unsupported consuming permutation scope")
            if controls and op not in {"X", "CX", "SWAP"}:
                reject(
                    token, "ownership", "coherent control cannot move or free storage"
                )
            if op == "allocate" and len(args) == 2:
                width = integer(args[0])
                if not 1 <= width <= 8:
                    reject(token, "width", "consuming allocation exceeds eight bits")
                bind(args[1], tuple(_Wire(0) for _ in range(width)))
            elif op == "free" and len(args) == 1:
                if any(wire.bit for wire in view(args[0])):
                    reject(
                        token,
                        "ownership",
                        "consuming free requires zero on every admitted input",
                    )
                live.pop(_atom(args[0]))
            elif op == "->" and len(args) == 2:
                source, target = args
                sources = source.children if source.token.text == "{" else (source,)
                targets = target.children if target.token.text == "{" else (target,)
                if len({_atom(node) for node in (*sources, *targets)}) != len(
                    sources
                ) + len(targets):
                    reject(token, "alias", "consuming binding names must be distinct")
                wires = tuple(wire for node in sources for wire in view(node))
                for node in sources:
                    live.pop(_atom(node))
                offset = 0
                for node in targets:
                    port = parameters.get(_atom(node))
                    destination_width = port.width if port is not None else None
                    if destination_width is None or offset + destination_width > len(
                        wires
                    ):
                        reject(
                            node.token,
                            "width",
                            "consuming bind needs an exact output partition",
                        )
                    bind(node, wires[offset : offset + destination_width])
                    offset += destination_width
                if offset != len(wires):
                    reject(
                        token, "width", "consuming bind must transfer every source wire"
                    )
            elif op in {"X", "CX", "SWAP"}:
                wires = tuple(scalar(arg) for arg in args)
                if len(wires) != (1 if op == "X" else 2) or len(set(wires)) != len(
                    wires
                ):
                    reject(
                        token, "alias", "consuming gates require distinct scalar wires"
                    )
                modified = wires if op == "SWAP" else wires[-1:]
                if controls.intersection(modified):
                    reject(
                        token, "alias", "a consuming gate cannot overwrite a control"
                    )
                if op == "X":
                    wires[0].bit ^= 1
                elif op == "CX":
                    wires[1].bit ^= wires[0].bit
                else:
                    wires[0].bit, wires[1].bit = wires[1].bit, wires[0].bit
            elif op in self.definitions:
                helper = self.definitions[op]
                if len(args) != len(helper.parameters) or len(
                    {_atom(arg) for arg in args}
                ) != len(args):
                    reject(token, "alias", "consuming call needs disjoint whole ports")
                children = {}
                outputs = {}
                for port, arg in zip(helper.parameters, args, strict=True):
                    label = _atom(arg)
                    if port.role == "output":
                        outputs[port.name.text] = arg
                        continue
                    bits = view(arg)
                    if port.role not in {"inout", "input"} or port.width != len(bits):
                        reject(
                            arg.token,
                            "ownership",
                            "consuming call port role or width disagrees",
                        )
                    children[port.name.text] = bits
                    if port.role == "input":
                        live.pop(label)
                for label, bits in self.invoke(helper, children, active).items():
                    bind(outputs[label], bits)
            else:
                reject(
                    token,
                    "ownership",
                    "consuming proof only admits reversible bit wiring",
                )


def _tokens(body: tuple[Statement, ...]) -> set[str]:
    from dataclasses import fields, is_dataclass

    stack: list[object] = list(body)
    result = set()
    while stack:
        node = stack.pop()
        if isinstance(node, Token):
            result.add(node.text)
        elif isinstance(node, tuple):
            stack.extend(node)
        elif is_dataclass(node):
            stack.extend(getattr(node, field.name) for field in fields(node))
    return result


def fuse_consuming_preparations(
    definitions: dict[str, Function],
) -> dict[str, Function]:
    consumed_functions = {
        name
        for name, function in definitions.items()
        if any(port.role == "input" for port in function.parameters)
    }
    if not consumed_functions:
        return definitions
    result = dict(definitions)
    direct = {name: _calls(function) for name, function in definitions.items()}
    excluded: set[str] = set()
    for name, function in definitions.items():
        statements = function.statements
        for start in range(len(statements) - 2):
            declaration, prepare = statements[start : start + 2]
            local = declaration.declaration
            if (
                local is None
                or local.width is None
                or not 1 <= local.width <= 6
                or prepare.token.text not in {"prepare_amplitudes", "prepare_state"}
                or len(prepare.arguments) != 3
                or prepare.arguments[2].children
                or prepare.arguments[2].token.text != local.name.text
            ):
                continue
            amplitudes, bound, _ = prepare.arguments
            if (
                amplitudes.token.text != "["
                or len(amplitudes.children) != 1 << local.width
            ):
                continue
            types = {port.name.text: port for port in function.parameters}
            types[local.name.text] = local
            locals_ = {local.name.text}
            preceding = local.name.text
            steps: list[Function] = []
            target: Parameter | None = None
            end = start + 2
            while end < len(statements):
                child = statements[end]
                end += 1
                if child.declaration is not None:
                    port = child.declaration
                    if port.name.text in types:
                        break
                    types[port.name.text] = port
                    locals_.add(port.name.text)
                    continue
                helper = definitions.get(child.token.text)
                if (
                    helper is None
                    or child.token.text not in consumed_functions
                    or len(helper.parameters) != 2
                    or tuple(port.role for port in helper.parameters)
                    != ("input", "output")
                    or len(child.arguments) != 2
                    or any(arg.children for arg in child.arguments)
                    or len({arg.token.text for arg in child.arguments}) != 2
                    or child.arguments[0].token.text != preceding
                    or types[preceding].width != helper.parameters[0].width
                ):
                    break
                preceding = child.arguments[1].token.text
                target = types.get(preceding)
                if target is None or target.width != helper.parameters[1].width:
                    break
                steps.append(helper)
                if target.role == "output":
                    break
            if (
                not steps
                or target is None
                or target.role != "output"
                or target.width is None
            ):
                continue
            if (
                target.width > 8
                or _tokens((*statements[:start], *statements[end:])) & locals_
            ):
                reject(
                    prepare.token,
                    "ownership",
                    "consuming preparation locals must not escape",
                )
            scope = set().union(
                *(_reachable(helper.name.text, direct) for helper in steps)
            )
            if any(
                direct[other] & scope for other in definitions.keys() - scope - {name}
            ):
                reject(
                    prepare.token,
                    "ownership",
                    "consuming preparation helpers must stay within the proof",
                )
            proof = _BasisWiring(definitions)
            values = [
                Expression(replace(prepare.token, text="0"))
                for _ in range(1 << target.width)
            ]
            occupied = set()
            for word, amplitude in enumerate(amplitudes.children):
                wires = tuple(_Wire(word >> bit & 1) for bit in range(local.width))
                for helper in steps:
                    source, output = helper.parameters
                    if source.width != len(wires):
                        reject(source.name, "width", "consuming input width disagrees")
                    wires = proof.invoke(helper, {source.name.text: wires})[
                        output.name.text
                    ]
                destination = sum(wire.bit << bit for bit, wire in enumerate(wires))
                if destination in occupied:
                    reject(
                        prepare.token,
                        "ownership",
                        "consuming basis mapping must be injective",
                    )
                occupied.add(destination)
                values[destination] = amplitude
            prepared = replace(
                prepare,
                arguments=(
                    replace(amplitudes, children=tuple(values)),
                    bound,
                    Expression(target.name),
                ),
            )
            result[name] = replace(
                function, statements=(*statements[:start], prepared, *statements[end:])
            )
            excluded.update(scope)
            break
    for name in excluded:
        result.pop(name)
    for name in consumed_functions - excluded:
        reject(
            definitions[name].name,
            "ownership",
            "consuming Input requires a checked preparation permutation",
        )
    return result
