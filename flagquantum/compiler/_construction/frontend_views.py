"""Bounded native quantum views, static loops and mutable gate-helper expansion."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import replace
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_classical import bind_helper
from .frontend_ports import _Port, _resolve
from .frontend_python import _error, _QuantumFunction
from .register_profile import _verify_borrowed_effects, register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def _require_builtin(owner: _Body, node: ast.Name, expected: object) -> None:
    function = owner.definition.function
    if not isinstance(function, FunctionType):
        _error(
            owner.source,
            node,
            "program.loop",
            "static builtins need a retained Python namespace",
        )
    visible = owner.aliases if owner.aliases is not None else owner.ports
    namespace = dict.get(function.__globals__, "__builtins__")
    builtin_values = vars(namespace) if type(namespace) is ModuleType else namespace
    builtin_value = (
        dict.get(builtin_values, node.id) if isinstance(builtin_values, dict) else None
    )
    resolved = (
        _resolve(owner.definition, node)
        if node.id in function.__code__.co_freevars
        else dict.get(function.__globals__, node.id, builtin_value)
    )
    if (
        node.id in visible
        or node.id in owner.loop_variables
        or node.id in owner.classical
        or resolved is not expected
    ):
        _error(
            owner.source, node, "program.loop", f"builtin {node.id} cannot be shadowed"
        )


def static_integer(owner: _Body, node: ast.expr, depth: int = 0) -> int:
    """Interpret the closed integer syntax without calling host Python."""
    if depth >= 32:
        _error(
            owner.source, node, "program.loop", "static integer expression is too deep"
        )
    value: object = None
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name):
        if node.id in owner.classical and type(owner.classical[node.id]) is not int:
            _error(
                owner.source,
                node,
                "program.loop",
                "real parameters cannot be static integers",
            )
        value = owner.classical.get(node.id, owner.loop_variables.get(node.id))
        if value is None and node.id not in (
            owner.aliases if owner.aliases is not None else owner.ports
        ):
            value = _resolve(owner.definition, node)
    elif isinstance(node, ast.Subscript):
        from .frontend_classical import classical_number

        value = classical_number(owner, node, depth + 1)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = static_integer(owner, node.operand, depth + 1)
        if isinstance(node.op, ast.USub):
            value = -value
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = static_integer(owner, node.left, depth + 1)
        right = static_integer(owner, node.right, depth + 1)
        value = (
            left + right
            if isinstance(node.op, ast.Add)
            else left - right if isinstance(node.op, ast.Sub) else left * right
        )
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
    ):
        _require_builtin(owner, node.func, len)
        argument = node.args[0]
        if isinstance(argument, ast.Name) and argument.id in owner.classical:
            array = owner.classical[argument.id]
            if not isinstance(array, tuple):
                _error(owner.source, node, "program.binding", "len requires an array")
            value = len(array)
        else:
            value = len(locations(owner, argument, depth + 1))
    if type(value) is not int or abs(value) > 4096:
        _error(
            owner.source,
            node,
            "program.loop",
            "expected a static integer in -4096..4096",
        )
    return value


def packed_locations(owner: _Body, node: ast.expr) -> tuple[tuple[Value, int], ...]:
    """Flatten an explicit storage sequence in logical least-significant order."""
    if isinstance(node, (ast.List, ast.Tuple)):
        if not 1 <= len(node.elts) <= 8:
            _error(
                owner.source,
                node,
                "program.view",
                "use one to eight explicit storage elements",
            )
        if any(isinstance(item, (ast.List, ast.Tuple)) for item in node.elts):
            _error(
                owner.source,
                node,
                "program.view",
                "nested storage sequences are not admitted",
            )
        return tuple(
            location for item in node.elts for location in locations(owner, item)
        )
    return locations(owner, node)


def locations(
    owner: _Body, node: ast.expr, depth: int = 0
) -> tuple[tuple[Value, int], ...]:
    """Resolve an ordered array view; element zero is the least-significant bit."""
    if depth >= 32:
        _error(owner.source, node, "program.view", "view nesting exceeds 32")
    if isinstance(node, ast.Name):
        if owner.aliases is not None:
            if node.id not in owner.aliases:
                _error(
                    owner.source,
                    node,
                    "program.view",
                    "mutable helpers access declared formal ports only",
                )
            return owner.aliases[node.id]
        value = owner.register(node)
        return tuple((value, index) for index in range(register_width(value)))
    if isinstance(node, ast.Subscript):
        base = node.value
        while isinstance(base, ast.Subscript):
            if not isinstance(base.slice, ast.Slice):
                _error(
                    owner.source,
                    node,
                    "program.view",
                    "a scalar element cannot be indexed again",
                )
            base = base.value
        if not isinstance(base, ast.Name) or base.id not in owner.array_names:
            _error(
                owner.source,
                node,
                "program.view",
                "index or slice a declared QArray[QBit]",
            )
        view = locations(owner, node.value, depth + 1)
        if isinstance(node.slice, ast.Slice):
            bounds = node.slice
            start = 0 if bounds.lower is None else static_integer(owner, bounds.lower)
            stop = (
                len(view)
                if bounds.upper is None
                else static_integer(owner, bounds.upper)
            )
            step = 1 if bounds.step is None else static_integer(owner, bounds.step)
            if step != 1 or not 0 <= start < stop <= len(view):
                _error(
                    owner.source,
                    node,
                    "program.view",
                    "use a nonempty in-bounds slice with step one",
                )
            return view[start:stop]
        index = static_integer(owner, node.slice)
        if not 0 <= index < len(view):
            _error(
                owner.source,
                node,
                "program.view",
                "quantum array index is out of bounds",
            )
        return (view[index],)
    _error(
        owner.source,
        node,
        "program.view",
        "use a quantum port or bounded static array view",
    )


def gate_target(owner: _Body, node: ast.expr) -> tuple[Value, int]:
    """Resolve one scalar gate target without allocating or copying storage."""
    view = locations(owner, node)
    if len(view) != 1:
        _error(
            owner.source,
            node,
            "program.width",
            "a scalar gate requires exactly one bit",
        )
    return view[0]


def capture_loop(owner: _Body, node: ast.For) -> None:
    """Unroll a builtin range within shared operation and iteration limits."""
    if any(
        isinstance(child, (ast.Return, ast.Break, ast.Continue))
        for child in ast.walk(node)
    ):
        _error(
            owner.source,
            node,
            "program.loop",
            "loop return, break and continue are not admitted",
        )
    visible = owner.aliases if owner.aliases is not None else owner.ports
    if (
        not isinstance(node.target, ast.Name)
        or node.target.id in visible
        or node.target.id in owner.loop_variables
        or node.target.id in owner.classical
        or node.target.id in {"range", "len"}
        or not isinstance(node.iter, ast.Call)
        or not isinstance(node.iter.func, ast.Name)
        or node.iter.func.id != "range"
        or not 1 <= len(node.iter.args) <= 3
        or node.iter.keywords
        or node.orelse
    ):
        _error(
            owner.source,
            node,
            "program.loop",
            "use a fresh index and builtin range with 1..3 static integer arguments",
        )
    _require_builtin(owner, node.iter.func, range)
    args = tuple(static_integer(owner, arg) for arg in node.iter.args)
    if len(args) == 3 and args[2] == 0:
        _error(owner.source, node, "program.loop", "range step cannot be zero")
    sequence = range(*args)
    if len(sequence) > 256:
        _error(owner.source, node, "program.loop", "one range exceeds 256 iterations")
    if len(owner.loop_variables) >= 8:
        _error(owner.source, node, "program.loop", "loop nesting exceeds eight")
    from .frontend_repeated_effects import capture_reset_loop

    if capture_reset_loop(owner, node, sequence):
        return
    expand_loop(owner, node, sequence)


def expand_loop(owner: _Body, node: ast.For, sequence: range) -> None:
    """Expand checked static iterations and charge their complete construction work."""
    assert isinstance(node.target, ast.Name)
    try:
        for index in sequence:
            owner.owner.loop_steps += 1
            if owner.owner.loop_steps > 4096:
                _error(
                    owner.source,
                    node,
                    "budget",
                    "program exceeds 4096 expanded loop iterations",
                )
            owner.loop_variables[node.target.id] = index
            owner.statements(node.body, nested=True)
    finally:
        owner.loop_variables.pop(node.target.id, None)


def inline_mutable(
    owner: _Body, node: ast.Call, definition: _QuantumFunction, ports: Sequence[_Port]
) -> None:
    """Expand checked borrowed helpers into the caller's owned effects."""
    from .frontend_evolution import flush_evolutions
    from .frontend_increment import is_increment_helper

    if any(port.role == "output" for port in ports):
        _error(
            owner.source,
            node,
            "program.borrow",
            "mutable helpers borrow Const or plain quantum ports without Outputs",
        )
    if id(definition) in owner.owner.active or len(owner.owner.active) >= 32:
        _error(
            owner.source,
            node,
            "program.recursion",
            "helper calls must be acyclic with depth at most 32",
        )
    owner.owner.inline_calls += 1
    if owner.owner.inline_calls > 256:
        _error(
            owner.source, node, "budget", "mutable helper expansion exceeds 256 calls"
        )
    if id(definition) not in owner.owner.seen:
        owner.owner.seen.add(id(definition))
        owner.owner.nodes += sum(1 for _ in ast.walk(definition.source.node))
        if owner.owner.nodes > 4096:
            _error(
                owner.source,
                node,
                "budget",
                "program definitions exceed 4096 AST nodes",
            )
    classical, arguments = bind_helper(owner, node, definition)
    bindings = {
        port.name: locations(owner, argument)
        for port, argument in zip(ports, arguments, strict=True)
    }
    increment_helper = is_increment_helper(definition)
    specialized = []
    for port in ports:
        view = bindings[port.name]
        actual = next(
            owner.ports[name]
            for name, value in owner.registers.items()
            if value == view[0][0]
        )
        whole = view == tuple(
            (view[0][0], bit) for bit in range(register_width(view[0][0]))
        )
        if whole and increment_helper and port.kind == "qnum" and actual.kind != "qnum":
            _error(
                owner.source,
                node,
                "program.kind",
                "QNum helpers require a QNum actual; use a writable raw unsigned QNum",
            )
        if port.width is not None and port.width != len(view):
            _error(
                owner.source,
                node,
                "program.width",
                "helper formal and actual widths disagree",
            )
        if port.kind != "qarray" and whole:
            if port.width is not None and (port.signed, port.fractional_bits) != (
                actual.signed,
                actual.fractional_bits,
            ):
                _error(
                    owner.source,
                    node,
                    "program.numeric.format",
                    "declared and actual helper numeric formats must agree",
                )
            port = replace(
                port,
                width=len(view),
                signed=actual.signed,
                fractional_bits=actual.fractional_bits,
            )
        specialized.append(port)
    ports = tuple(specialized)
    occupied: set[tuple[Value, int]] = set()
    readonly = {
        owner.registers[p.name]
        for p in owner.ports.values()
        if p.role == "const" and p.name in owner.registers
    }
    for port in ports:
        view = bindings[port.name]
        if (port.kind == "qbit" and len(view) != 1) or (
            port.width is not None and port.width != len(view)
        ):
            _error(
                owner.source,
                node,
                "program.width",
                "helper formal and actual widths disagree",
            )
        if port.role == "inout" and any(value in readonly for value, _ in view):
            _error(
                owner.source,
                node,
                "program.borrow",
                "Const cannot be passed as writable helper storage",
            )
        if occupied.intersection(view):
            _error(
                owner.source,
                node,
                "program.borrow",
                "mutable helper arguments must be disjoint writable locations",
            )
        occupied.update(view)
    const_values = tuple(
        dict.fromkeys(
            value
            for port in ports
            if port.role == "const"
            for value, _ in bindings[port.name]
        )
    )
    writable_values = tuple(
        dict.fromkeys(
            value
            for port in ports
            if port.role == "inout"
            for value, _ in bindings[port.name]
        )
    )
    if set(const_values).intersection(writable_values):
        _error(
            owner.source,
            node,
            "program.borrow",
            "Const and writable helper views must use separate registers",
        )
    if all(
        view == tuple((view[0][0], bit) for bit in range(register_width(view[0][0])))
        for view in bindings.values()
    ):
        # Whole-register functions can use the ordinary capture context, including
        # fresh lexical scratch, without leaking formal names into caller state.
        from .frontend_program import _Body

        concrete = tuple(
            replace(port, width=len(bindings[port.name])) for port in ports
        )
        child = _Body(
            owner.owner,
            definition,
            f"{owner.name}.borrow{owner.owner.inline_calls}",
            concrete,
            classical,
        )
        child.registers = {port.name: bindings[port.name][0][0] for port in ports}
        child.effect, child.counter = owner.effect, owner.counter
        storage = dict(child.registers), dict(child.ports), dict(child.widths)
        owner.owner.active.add(id(definition))
        try:
            child.statements(definition.source.node.body)
            flush_evolutions(child)
        finally:
            owner.owner.active.remove(id(definition))
        if (
            storage != (child.registers, child.ports, child.widths)
            or child.hidden
            or child.dropped
        ):
            _error(
                owner.source,
                node,
                "program.borrow",
                "borrowed helpers must close all local storage without ownership changes",
            )
        _verify_borrowed_effects(
            child.operations, owner.owner.functions, const_values, writable_values
        )
        owner.operations.extend(child.operations)
        owner.effect, owner.counter = child.effect, child.counter
        return
    start = len(owner.operations)
    saved = (
        owner.definition,
        owner.source,
        owner.aliases,
        owner.alias_ports,
        owner.array_names,
        owner.loop_variables,
        owner.classical,
    )
    owner.definition, owner.source = definition, definition.source
    owner.aliases = bindings
    owner.alias_ports = {port.name: port for port in ports}
    owner.array_names = {port.name for port in ports if port.kind == "qarray"}
    owner.loop_variables = {}
    owner.classical = classical
    owner.owner.active.add(id(definition))
    try:
        owner.statements(definition.source.node.body)
        flush_evolutions(owner)
        if const_values:
            _verify_borrowed_effects(
                owner.operations[start:],
                owner.owner.functions,
                const_values,
                writable_values,
            )
    finally:
        owner.owner.active.remove(id(definition))
        (
            owner.definition,
            owner.source,
            owner.aliases,
            owner.alias_ports,
            owner.array_names,
            owner.loop_variables,
            owner.classical,
        ) = saved
