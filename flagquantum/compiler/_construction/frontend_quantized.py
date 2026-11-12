"""Capture closed precision-annotated Python on shared bounded numeric facts."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from functools import partial
from typing import NoReturn

from .._hybrid.model import HybridProgram, Value
from .._hybrid.verifier import HybridVerificationError
from .fixed_point import (
    FixedPointBranches,
    FixedPointExpression,
    FixedPointScalar,
)
from .fixed_point_program import FixedPointScope, emit_fixed_point_tables
from .frontend_ports import (
    _annotation_node,
    _Port,
    _resolve,
    classical_parameters,
    program_ports,
    quantum_port,
)
from .frontend_python import _error, _QuantumFunction
from .program_types import (
    QBit,
    QNum,
    allocate,
    drop,
    hadamard_transform,
    prepare_state,
    quantize,
)


def _reachable_marker(definition: _QuantumFunction) -> bool:
    seen: set[int] = set()
    pending = [definition]
    while pending and len(seen) <= 32:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for node in ast.walk(current.source.node):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, (ast.Name, ast.Attribute)
            ):
                continue
            try:
                function = _resolve(current, node.func)
            except HybridVerificationError:
                # Ordinary capture reports dynamic/unresolved calls. Detection
                # never evaluates an object or admits an unsupported call.
                continue
            if function is quantize:
                return True
            if isinstance(function, _QuantumFunction):
                pending.append(function)
    return False


class _Native:
    def __init__(self) -> None:
        self.branches = FixedPointBranches()
        self.steps = 0
        self.nodes = 0
        self.seen: set[int] = set()
        self.active: set[int] = set()

    def function(
        self,
        definition: _QuantumFunction,
        inputs: Mapping[str, FixedPointScalar],
        *,
        root: bool = False,
    ) -> tuple[tuple[_Port, ...], tuple[FixedPointScalar, ...]]:
        source = definition.source

        def fail(node: ast.expr | ast.stmt, code: str, message: str) -> NoReturn:
            _error(source, node, f"program.quantized.{code}", message)

        if id(definition) in self.active or len(self.active) >= 12:
            fail(
                source.node,
                "budget",
                "quantized helpers must be acyclic with depth below twelve",
            )
        if id(definition) not in self.seen:
            self.seen.add(id(definition))
            self.nodes += sum(1 for _ in ast.walk(source.node))
        if self.nodes > 4096 or len(self.seen) > 32:
            fail(
                source.node,
                "budget",
                "quantized definitions exceed the AST/function budget",
            )
        ports = program_ports(definition)
        if (
            not ports
            or classical_parameters(definition)
            or any(p.kind not in {"qnum", "qbit"} or p.role == "inout" for p in ports)
            or root
            and any(p.role != "output" for p in ports)
        ):
            fail(
                source.node,
                "ports",
                "quantized programs require numeric Output/Const ports and an Output-only root",
            )
        scope = FixedPointScope(self.branches, {p.name: p.role for p in ports}, inputs)
        declarations = {p.name: p for p in ports}

        def name(node: ast.expr) -> str:
            if not isinstance(node, ast.Name) or node.id not in scope.roles:
                fail(node, "ownership", "use one declared whole numeric variable")
            return node.id

        def checked(
            port: _Port, value: FixedPointScalar, node: ast.expr | ast.stmt
        ) -> None:
            if port.width is not None and (
                port.width,
                port.signed,
                port.fractional_bits,
            ) != (value.width, value.signed, value.fraction):
                fail(
                    node, "format", "declared and inferred quantized formats must agree"
                )

        for port in ports:
            if port.role == "const":
                checked(
                    port,
                    scope.read(port.name, lambda c, m: fail(source.node, c, m)),
                    source.node,
                )

        def literal(node: ast.expr) -> Fraction | None:
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and type(node.value) in {int, float}
            ):
                if abs(node.value) > 2**20 or not math.isfinite(node.value):
                    fail(
                        node,
                        "precision",
                        "numeric literal exceeds the bounded fixed-point range",
                    )
                return Fraction(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                value = literal(node.operand)
                return None if value is None else -value
            return None

        def text(node: ast.expr) -> str:
            if isinstance(node, ast.Name):
                target = name(node)
                scope.read(target, lambda c, m: fail(node, c, m))
                return target
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Pow) and (
                    not isinstance(node.right, ast.Constant)
                    or type(node.right.value) is not int
                ):
                    fail(
                        node, "power", "numeric powers require a literal integer degree"
                    )
                return {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Pow: "**"}.get(
                    type(node.op), "unsupported"
                )
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return "-"
            return "unsupported"

        def children(node: ast.expr) -> tuple[ast.expr, ...]:
            if isinstance(node, ast.BinOp):
                return node.left, node.right
            if isinstance(node, ast.UnaryOp):
                return (node.operand,)
            return ()

        evaluator = FixedPointExpression(
            text=text, children=children, literal=literal, reject=fail
        )

        def resolve(node: ast.expr) -> object:
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in scope.roles:
                fail(node, "call", "a numeric variable cannot shadow a callable")
            return _resolve(definition, node)

        def bind(
            destination: str, value: FixedPointScalar, node: ast.expr | ast.stmt
        ) -> None:
            checked(declarations[destination], value, node)
            scope.values[destination] = value

        def precision(node: ast.expr, expected: int) -> bool:
            return (
                isinstance(node, ast.Constant)
                and type(node.value) is int
                and node.value == expected
            )

        def assignment(node: ast.AugAssign) -> None:
            pending: list[tuple[ast.AST, int]] = [(node.value, 0)]
            count = 0
            while pending:
                child, depth = pending.pop()
                count += 1
                if count > 512 or depth >= 128:
                    fail(
                        node,
                        "budget",
                        "quantized expression exceeds 512 nodes/128 depth",
                    )
                pending.extend(
                    (item, depth + 1) for item in ast.iter_child_nodes(child)
                )
            if not isinstance(node.op, ast.BitOr):
                fail(
                    node,
                    "assignment",
                    "quantized expressions initialize fresh storage with |= only",
                )
            destination = scope.reserve(
                name(node.target), lambda c, m: fail(node.target, c, m)
            )
            expr = node.value
            if isinstance(expr, ast.Call) and resolve(expr.func) is quantize:
                keywords = {key.arg: key.value for key in expr.keywords}
                if (
                    len(expr.args) != 1
                    or len(expr.keywords) != 2
                    or set(keywords) != {"fractional_bits", "literal_bits"}
                    or any(
                        not precision(keywords[key], value)
                        for key, value in (("fractional_bits", 8), ("literal_bits", 20))
                    )
                ):
                    fail(
                        expr,
                        "precision",
                        "quantize requires one expression and literal fractional_bits=8, literal_bits=20",
                    )
                value = evaluator.evaluate(expr.args[0], scope.values)
            else:
                value = evaluator.evaluate(expr, scope.values, quantized=False)
            bind(destination, value, node)

        def call(node: ast.Call) -> None:
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                fail(
                    node,
                    "signature",
                    "quantized calls use positional arguments without unpacking",
                )
            function = resolve(node.func)
            reject = partial(fail, node)
            if function is allocate and len(node.args) in {1, 2}:
                destination = name(node.args[-1])
                size = node.args[0] if len(node.args) == 2 else None
                width = declarations[destination].width
                if size is not None:
                    if (
                        not isinstance(size, ast.Constant)
                        or type(size.value) is not int
                    ):
                        fail(
                            size,
                            "width",
                            "quantized allocation requires literal width one",
                        )
                    width = size.value
                if width != 1:
                    fail(
                        node,
                        "budget",
                        "independently prepared quantized registers require one bit",
                    )
                scope.allocate(destination, reject)
                checked(declarations[destination], scope.values[destination], node)
            elif function is hadamard_transform and len(node.args) == 1:
                scope.hadamard(name(node.args[0]), reject)
            elif function is prepare_state and len(node.args) == 3:
                vector, bound, target = node.args
                if (
                    not isinstance(vector, (ast.List, ast.Tuple))
                    or tuple(literal(item) for item in vector.elts)
                    != (Fraction(1, 2), Fraction(1, 2))
                    or literal(bound) != 0
                ):
                    fail(
                        node,
                        "prepare",
                        "bounded preparation requires probabilities [0.5, 0.5] and zero error",
                    )
                destination = name(target)
                scope.prepare_uniform(destination, reject)
                checked(declarations[destination], scope.values[destination], node)
            elif function is drop and len(node.args) == 1:
                scope.drop(name(node.args[0]), reject)
            elif isinstance(function, _QuantumFunction) and function.source.program:
                formal = program_ports(function)
                if len(node.args) != len(formal):
                    fail(node, "signature", "qfunc argument count disagrees")
                actual = tuple(name(arg) for arg in node.args)
                if len(set(actual)) != len(actual):
                    fail(
                        node,
                        "alias",
                        "quantized helper actuals must be disjoint whole words",
                    )
                child_inputs = {}
                writes = []
                for port, destination in zip(formal, actual, strict=True):
                    if port.kind != declarations[destination].kind:
                        fail(
                            node, "format", "actual and formal numeric kinds must agree"
                        )
                    if port.role == "output":
                        writes.append(scope.reserve(destination, reject))
                    elif port.role == "const":
                        child_inputs[port.name] = scope.read(destination, reject)
                    else:
                        fail(
                            node, "borrow", "quantized helper reads require Const ports"
                        )
                _, outputs = self.function(function, child_inputs)
                for destination, value in zip(writes, outputs, strict=True):
                    bind(destination, value, node)
            else:
                fail(
                    node,
                    "scope",
                    "unsupported effect in a closed quantized numeric program",
                )

        self.active.add(id(definition))
        try:
            for statement in source.node.body:
                self.steps += 1
                if self.steps > 256:
                    fail(
                        statement,
                        "budget",
                        "quantized expansion exceeds 256 statements",
                    )
                if isinstance(statement, ast.AnnAssign):
                    if not isinstance(statement.target, ast.Name):
                        fail(statement, "ownership", "declare one typed numeric local")
                    annotation = _annotation_node(source, statement.annotation)
                    port = quantum_port(
                        definition, statement.target.id, annotation, annotation, "inout"
                    )
                    if port.kind not in {"qnum", "qbit"}:
                        fail(
                            statement,
                            "format",
                            "quantized locals require scalar numeric types",
                        )
                    if statement.value is not None:
                        value = statement.value
                        if (
                            not isinstance(value, ast.Call)
                            or value.args
                            or value.keywords
                            or resolve(value.func)
                            is not (QBit if port.kind == "qbit" else QNum)
                        ):
                            fail(
                                statement,
                                "ownership",
                                "declare a local with its matching empty quantum constructor",
                            )
                    scope.declare(port.name, partial(fail, statement))
                    declarations[port.name] = port
                elif isinstance(statement, ast.AugAssign):
                    assignment(statement)
                elif isinstance(statement, ast.Expr) and isinstance(
                    statement.value, ast.Call
                ):
                    call(statement.value)
                elif (
                    isinstance(statement, ast.Return)
                    and (
                        statement.value is None
                        or isinstance(statement.value, ast.Constant)
                        and statement.value.value is None
                    )
                ) or (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    continue
                else:
                    fail(
                        statement,
                        "scope",
                        "quantized capture requires an unconditional straight-line program",
                    )
            missing = {
                name for name, role in scope.roles.items() if role == "local"
            } - scope.retired
            if missing:
                fail(
                    source.node,
                    "ownership",
                    "drop every owned numeric local before return: "
                    + ", ".join(sorted(missing)),
                )
            outputs = tuple(
                scope.read(p.name, lambda c, m: fail(source.node, c, m))
                for p in ports
                if p.role == "output"
            )
            resolved = tuple(
                replace(
                    p,
                    width=scope.values[p.name].width,
                    signed=scope.values[p.name].signed,
                    fractional_bits=scope.values[p.name].fraction,
                )
                for p in ports
            )
            return resolved, outputs
        finally:
            self.active.remove(id(definition))


def try_capture_quantized(definition: _QuantumFunction) -> HybridProgram | None:
    if not _reachable_marker(definition):
        return None
    planner = _Native()
    ports, outputs = planner.function(definition, {}, root=True)
    planner.branches.verify_discarded(
        lambda c, m: _error(
            definition.source, definition.source.node, f"program.quantized.{c}", m
        )
    )
    return _emit(definition, ports, outputs, planner.branches.bits)


def _emit(
    definition: _QuantumFunction,
    ports: tuple[_Port, ...],
    outputs: tuple[FixedPointScalar, ...],
    bits: int,
) -> HybridProgram:
    # Reuse native declarations, numeric formats, source locations and effect
    # construction; no source statements are replayed or host functions called.
    from .frontend_program import _Body, _Capture

    name = definition.source.node.name
    owner = _Body(_Capture(definition, {}), definition, name, ports)
    node = definition.source.node
    reads: tuple[tuple[Value, int], ...] = ()
    if bits:
        selector_name = "$quantized_branch"
        owner.ports[selector_name] = _Port(selector_name, "inout", "qnum", bits)
        owner.widths[selector_name] = bits
        owner.hidden.add(selector_name)
        selector = owner.register(ast.copy_location(ast.Name(id=selector_name), node))
        owner.emit("qreg.allocate", node, (selector,))
        reads = tuple((selector, bit) for bit in range(bits))
        for _, bit in reads:
            owner.emit("qreg.gate", node, (selector,), gate="h", index=bit)
    targets = []
    for port, value in zip(ports, outputs, strict=True):
        target = owner.register(ast.copy_location(ast.Name(id=port.name), node))
        owner.emit("qreg.allocate", node, (target,))
        targets.append((value, tuple((target, bit) for bit in range(value.width))))

    def flip(target: tuple[Value, int]) -> None:
        owner.emit("qreg.gate", node, (target[0],), gate="x", index=target[1])

    def controlled(
        controls: tuple[tuple[Value, int], ...], pattern: int, target: tuple[Value, int]
    ) -> None:
        combined = (*controls, target)
        owner.emit(
            "qreg.control_x",
            node,
            tuple(value for value, _ in combined),
            indices=tuple(bit for _, bit in combined),
            control_state=pattern,
        )

    emit_fixed_point_tables(reads, targets, flip=flip, controlled=controlled)
    if bits:
        owner.emit("qreg.drop", node, (selector,))
        owner.dropped.add(selector_name)
    return owner.program(numeric_formats=True)
