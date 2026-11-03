"""Capture typed Python programs into existing checked register/effect records."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from types import FunctionType, ModuleType
from typing import Literal

from .._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from .frontend_python import _error, _QuantumFunction, _Source, capture
from .profile import _width
from .program_types import (
    Const,
    Output,
    QBit,
    QNum,
    _width_spec,
    adjoint,
    allocate,
    apply,
    crx,
    cx,
    hadamard_transform,
    phase,
    r,
    swap,
    within,
)
from .ranges import unsigned_width
from .register_profile import (
    equatorial_rotation,
    register_type,
    register_width,
    verify_register_program,
)


@dataclass(frozen=True)
class _Port:
    name: str
    role: Literal["output", "const"]
    kind: Literal["qbit", "qnum"]
    width: int | None


def _resolve(
    definition: _QuantumFunction,
    node: ast.expr,
    depth: int = 0,
    *,
    annotation: bool = False,
) -> object:
    source = definition.source
    function = definition.function
    if depth >= 32:
        _error(source, node, "budget", "qualified symbol depth exceeds 32")
    if not isinstance(function, FunctionType):
        _error(
            source,
            node,
            "annotation",
            "program capture requires a retained Python namespace",
        )
    if isinstance(node, ast.Name):
        if (
            annotation
            and definition.annotation_bindings is not None
            and node.id in definition.annotation_bindings
        ):
            return definition.annotation_bindings[node.id]
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        ):
            if name == node.id:
                try:
                    return cell.cell_contents
                except ValueError:
                    _error(
                        source, node, "name", "the referenced closure binding is empty"
                    )
        return function.__globals__.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _resolve(definition, node.value, depth + 1, annotation=annotation)
        if type(parent) is ModuleType:
            return vars(parent).get(node.attr)
    _error(
        source,
        node,
        "name",
        "use a static imported symbol, not dynamic attribute access",
    )


def _annotation_node(source: _Source, node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            parsed = ast.parse(node.value, mode="eval").body
        except (SyntaxError, RecursionError):
            _error(source, node, "annotation", "invalid quoted quantum annotation")
        if sum(1 for _ in ast.walk(parsed)) > 4096:
            _error(
                source, node, "budget", "quantum annotation exceeds the source node cap"
            )
        ast.increment_lineno(parsed, node.lineno - 1)
        return parsed
    return node


def program_annotation_names(source: _Source) -> set[str]:
    return {
        node.id
        for arg in (*source.node.args.posonlyargs, *source.node.args.args)
        if arg.annotation is not None
        for node in ast.walk(_annotation_node(source, arg.annotation))
        if isinstance(node, ast.Name)
    }


def program_ports(definition: _QuantumFunction) -> tuple[_Port, ...]:
    source = definition.source
    ports: list[_Port] = []
    for argument in (*source.node.args.posonlyargs, *source.node.args.args):
        annotation = argument.annotation
        if annotation is None:
            _error(
                source,
                source.node,
                "annotation",
                "program ports require Output or Const annotations",
            )
        node = _annotation_node(source, annotation)
        if not isinstance(node, ast.Subscript):
            _error(
                source,
                annotation,
                "annotation",
                "use Output[QNum], Const[QNum] or a sized QNum/QBit port",
            )
        role = _resolve(definition, node.value, annotation=True)
        if role is not Output and role is not Const:
            _error(
                source,
                annotation,
                "annotation",
                "only Output and Const program roles are admitted",
            )
        quantum = node.slice
        width = None
        if isinstance(quantum, ast.Subscript):
            kind = _resolve(definition, quantum.value, annotation=True)
            if kind is not QNum:
                _error(
                    source,
                    annotation,
                    "annotation",
                    "only QNum accepts numeric type parameters",
                )
            sizes = (
                quantum.slice.elts
                if isinstance(quantum.slice, ast.Tuple)
                else (quantum.slice,)
            )
            if any(not isinstance(size, ast.Constant) for size in sizes):
                _error(
                    source,
                    annotation,
                    "annotation",
                    "numeric widths/sign/fraction must be literals",
                )
            spec = tuple(size.value for size in sizes if isinstance(size, ast.Constant))
            try:
                width = _width_spec(spec)
            except (TypeError, ValueError) as error:
                _error(source, annotation, "annotation", str(error))
        else:
            kind = _resolve(definition, quantum, annotation=True)
        if kind is not QNum and kind is not QBit:
            _error(
                source, annotation, "annotation", "program ports require QNum or QBit"
            )
        ports.append(
            _Port(
                argument.arg,
                "output" if role is Output else "const",
                "qbit" if kind is QBit else "qnum",
                1 if kind is QBit else width,
            )
        )
    returned = source.node.returns
    if returned is not None and not (
        isinstance(returned, ast.Constant) and returned.value in (None, "None")
    ):
        _error(
            source,
            returned,
            "annotation",
            "program outputs use signature ports; return annotation is None",
        )
    statements = source.node.body
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    for index, statement in enumerate(statements):
        if (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.op, ast.BitOr)
            and isinstance(statement.target, ast.Name)
        ):
            continue
        if isinstance(statement, ast.With):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            continue
        if (
            isinstance(statement, ast.Return)
            and index == len(statements) - 1
            and (
                statement.value is None
                or isinstance(statement.value, ast.Constant)
                and statement.value.value is None
            )
        ):
            continue
        _error(
            source,
            statement,
            "program.statement",
            "use allocate, hadamard_transform, fresh |= arithmetic and static qfunc calls",
        )
    return tuple(ports)


class _Capture:
    def __init__(self, entry: _QuantumFunction) -> None:
        self.entry = entry.source.node.name
        self.functions: list[HybridProgram] = []
        self.active: set[int] = set()
        self.seen: set[int] = set()
        self.nodes = 0
        self.serial = 0
        self.cache: dict[
            tuple[int, tuple[int | None, ...]], tuple[HybridProgram, tuple[_Port, ...]]
        ] = {}

    def build(
        self,
        definition: _QuantumFunction,
        widths: tuple[int | None, ...] = (),
        *,
        root: bool = False,
    ) -> tuple[HybridProgram, tuple[_Port, ...]]:
        source = definition.source
        if id(definition) in self.active:
            _error(
                source, source.node, "program.recursion", "qfunc calls must be acyclic"
            )
        key = (id(definition), widths)
        if key in self.cache:
            return self.cache[key]
        if self.serial >= 32 or len(self.active) >= 32:
            _error(
                source,
                source.node,
                "budget",
                "program specialization/call depth exceeds 32",
            )
        if id(definition) not in self.seen:
            self.seen.add(id(definition))
            self.nodes += sum(1 for _ in ast.walk(source.node))
            if self.nodes > 4096:
                _error(
                    source,
                    source.node,
                    "budget",
                    "program definitions exceed 4096 AST nodes",
                )
        ports = program_ports(definition)
        if root and (not ports or any(p.role != "output" for p in ports)):
            _error(
                source,
                source.node,
                "program.entry",
                "a program root requires Output-only ports",
            )
        if widths:
            if len(widths) != len(ports):
                _error(
                    source,
                    source.node,
                    "program.signature",
                    "qfunc argument count disagrees",
                )
            specialized = []
            for port, width in zip(ports, widths, strict=True):
                if port.width is not None and width is not None and port.width != width:
                    _error(
                        source,
                        source.node,
                        "program.width",
                        "declared and actual widths must agree",
                    )
                specialized.append(
                    replace(port, width=port.width if port.width is not None else width)
                )
            ports = tuple(specialized)
        name = (
            self.entry if root else f"{self.entry}.call{self.serial}.{source.node.name}"
        )
        self.serial += 1
        self.active.add(id(definition))
        try:
            builder = _Body(self, definition, name, ports)
            program = builder.finish()
            resolved = tuple(replace(p, width=builder.widths[p.name]) for p in ports)
        finally:
            self.active.remove(id(definition))
        self.cache[key] = (program, resolved)
        if not root:
            self.functions.append(program)
        return program, resolved


class _Body:
    def __init__(
        self,
        owner: _Capture,
        definition: _QuantumFunction,
        name: str,
        ports: tuple[_Port, ...],
    ) -> None:
        self.owner, self.definition, self.name = owner, definition, name
        self.source = definition.source
        self.ports = {p.name: p for p in ports}
        self.widths = {p.name: p.width for p in ports}
        self.registers: dict[str, Value] = {}
        self.entry = Value(ValueId(name, 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.counter = len(ports) + 1
        self.operations: list[Operation] = []
        self.expressions = 0

    def register(self, node: ast.expr, inferred: int | None = None) -> Value:
        if not isinstance(node, ast.Name) or node.id not in self.ports:
            _error(self.source, node, "program.port", "use one declared quantum port")
        name = node.id
        width = self.widths[name]
        if width is None:
            self.widths[name] = width = inferred
        if width is None:
            _error(
                self.source,
                node,
                "program.width",
                "declare a width or initialize the port first",
            )
        if inferred is not None and inferred > width:
            _error(
                self.source,
                node,
                "program.width",
                "output width cannot fit the complete expression domain",
            )
        if name not in self.registers:
            self.registers[name] = Value(
                ValueId(self.name, tuple(self.ports).index(name) + 1),
                register_type(width),
            )
        return self.registers[name]

    def emit(
        self,
        opcode: str,
        node: ast.expr | ast.stmt,
        values: Sequence[Value],
        **attributes: object,
    ) -> None:
        if len(self.operations) >= 4096:
            _error(
                self.source, node, "budget", "program exceeds 4096 effect operations"
            )
        following = Value(ValueId(self.name, self.counter), QUANTUM_EFFECT)
        self.counter += 1
        self.operations.append(
            Operation(
                opcode,
                (self.effect, *values),
                (following,),
                attributes,
                location=self.source.location(node),
            )
        )
        self.effect = following

    def assign(self, node: ast.AugAssign) -> None:
        expr = node.value
        if (
            isinstance(expr, ast.Constant)
            and type(expr.value) is int
            and expr.value >= 0
        ):
            width = unsigned_width(expr.value)
            if width > 256:
                _error(
                    self.source,
                    expr,
                    "program.width",
                    "integer literal exceeds 256 bits",
                )
            self.emit(
                "qreg.initialize",
                node,
                (self.register(node.target, width),),
                value=expr.value,
            )
            return
        graph, actuals = self.capture_expression(expr, node)
        width = _width(graph.body.blocks[0].operations[-1].operands[0].type, 256)
        self.emit(
            "qreg.compute",
            node,
            (self.register(node.target, width), *actuals),
            callee=graph.name,
        )

    def capture_expression(
        self, expr: ast.expr, statement: ast.AST
    ) -> tuple[HybridProgram, tuple[Value, ...]]:
        allowed = (
            ast.BinOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Mult,
            ast.Pow,
        )
        if any(not isinstance(child, allowed) for child in ast.walk(expr)):
            _error(
                self.source,
                expr,
                "program.expression",
                "use unsigned +, *, literal powers 1..8, or an unsigned literal",
            )
        used = {child.id for child in ast.walk(expr) if isinstance(child, ast.Name)}
        if not used or not used.issubset(self.ports):
            _error(
                self.source,
                expr,
                "program.expression",
                "arithmetic reads declared quantum ports; basis initialization uses an unsigned literal",
            )
        names = tuple(name for name in self.ports if name in used)
        actuals = tuple(
            self.register(ast.copy_location(ast.Name(id=name, ctx=ast.Load()), expr))
            for name in names
        )
        function = deepcopy(self.source.node)
        function.name = f"{self.name}.expr{self.expressions}"
        self.expressions += 1
        function.args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in names],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        function.decorator_list = []
        function.returns = None
        function.body = [ast.copy_location(ast.Return(value=deepcopy(expr)), statement)]
        graph = capture(
            _QuantumFunction(
                _Source(
                    function,
                    self.source.filename,
                    self.source.first_line,
                    self.source.indentation,
                )
            ),
            {name: register_width(self.registers[name]) for name in names},
        )
        self.owner.functions.append(graph)
        return graph, actuals

    def phase_number(self, node: ast.expr) -> float | None:
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        value: object
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, (ast.Name, ast.Attribute)):
            if isinstance(node, ast.Name) and node.id in self.ports:
                return None
            value = _resolve(self.definition, node)
        else:
            return None
        if type(value) is not int and type(value) is not float:
            return None
        try:
            converted = sign * float(value)
        except OverflowError:
            return None
        return converted if math.isfinite(converted) else None

    def call(self, node: ast.Call, *, inverse: bool = False) -> None:
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            _error(
                self.source,
                node,
                "program.signature",
                "use positional arguments without unpacking",
            )
        if isinstance(node.func, ast.Name) and node.func.id in self.ports:
            _error(
                self.source,
                node.func,
                "program.call",
                "a quantum port cannot shadow a callable",
            )
        function = _resolve(self.definition, node.func)
        if inverse and (
            not isinstance(function, _QuantumFunction)
            or not function.source.program
            or any(port.role != "const" for port in program_ports(function))
        ):
            _error(
                self.source,
                node,
                "program.adjoint",
                "adjoint requires a static Const-only qfunc",
            )
        if function is allocate:
            if len(node.args) not in {1, 2}:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "allocate accepts an output or literal size and output",
                )
            size = None
            if len(node.args) == 2:
                size_node = node.args[0]
                if (
                    not isinstance(size_node, ast.Constant)
                    or type(size_node.value) is not int
                ):
                    _error(
                        self.source,
                        size_node,
                        "program.width",
                        "allocation size must be a literal integer in 1..256",
                    )
                try:
                    size = _width_spec(size_node.value)
                except (TypeError, ValueError) as error:
                    _error(self.source, size_node, "program.width", str(error))
            target = node.args[-1]
            destination = self.register(target, size)
            if size is not None and register_width(destination) != size:
                _error(
                    self.source,
                    target,
                    "program.width",
                    "allocation and declared width must agree",
                )
            self.emit("qreg.allocate", node, (destination,))
        elif function is hadamard_transform:
            if len(node.args) != 1:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "hadamard_transform takes one quantum port",
                )
            destination = self.register(node.args[0])
            for bit in range(register_width(destination)):
                self.emit("qreg.gate", node, (destination,), gate="h", index=bit)
        elif function is cx or function is swap or function is crx:
            angled = function is crx
            if len(node.args) != 2 + angled:
                _error(
                    self.source, node, "program.signature", "invalid pair gate arity"
                )
            attributes: dict[str, str | float] = {"gate": function.__name__}
            if angled:
                angle = self.phase_number(node.args[0])
                if angle is None:
                    _error(
                        self.source,
                        node.args[0],
                        "program.angle",
                        "CRX requires a finite static real angle",
                    )
                attributes["angle"] = angle
            operands = tuple(self.register(arg) for arg in node.args[int(angled) :])
            self.emit("qreg.pair_gate", node, operands, **attributes)
        elif function is r:
            if len(node.args) != 3:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "R requires two angles and a target",
                )
            theta = self.phase_number(node.args[0])
            phi = self.phase_number(node.args[1])
            if theta is None or phi is None:
                _error(
                    self.source,
                    node,
                    "program.angle",
                    "R requires finite static real angles",
                )
            target = self.register(node.args[2])
            if register_width(target) != 1:
                _error(
                    self.source, node, "program.width", "R requires a one-bit target"
                )
            for gate, angle in equatorial_rotation(theta, phi):
                self.emit("qreg.gate", node, (target,), gate=gate, index=0, angle=angle)
        elif function is phase:
            if len(node.args) != 2:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "phase takes an expression and coefficient",
                )
            expr, coefficient = node.args
            angle = self.phase_number(coefficient)
            if angle is None:
                _error(
                    self.source,
                    coefficient,
                    "program.phase",
                    "phase coefficient must be a finite static real",
                )
            constant = self.phase_number(expr)
            if constant is not None:
                radians = constant * angle
                if not math.isfinite(radians):
                    _error(
                        self.source,
                        expr,
                        "program.phase",
                        "constant phase angle must be finite",
                    )
                self.emit("qreg.phase", node, (), angle=radians)
            else:
                graph, actuals = self.capture_expression(expr, node)
                width = _width(
                    graph.body.blocks[0].operations[-1].operands[0].type, 256
                )
                attributes = {"angle": angle} if angle != math.pi or width != 1 else {}
                self.emit("qreg.phase", node, actuals, callee=graph.name, **attributes)
        elif isinstance(function, _QuantumFunction) and function.source.program:
            formal = program_ports(function)
            if len(node.args) != len(formal):
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "qfunc argument count disagrees",
                )
            widths: list[int | None] = []
            for argument, port in zip(node.args, formal, strict=True):
                if not isinstance(argument, ast.Name) or argument.id not in self.ports:
                    _error(
                        self.source,
                        argument,
                        "program.port",
                        "qfunc actuals must be declared quantum ports",
                    )
                if self.ports[argument.id].kind != port.kind:
                    _error(
                        self.source,
                        argument,
                        "program.kind",
                        "qfunc actual and formal quantum kinds must agree",
                    )
                widths.append(self.widths[argument.id])
                if port.role == "const":
                    self.register(argument)
            program, resolved = self.owner.build(function, tuple(widths))
            reads: list[Value] = []
            writes: list[Value] = []
            for argument, port in zip(node.args, resolved, strict=True):
                (reads if port.role == "const" else writes).append(
                    self.register(argument, port.width)
                )
            call_attributes: dict[str, object] = {"callee": program.name}
            if inverse:
                call_attributes["inverse"] = True
            self.emit("qreg.call", node, (*reads, *writes), **call_attributes)
        else:
            _error(
                self.source,
                node.func,
                "program.call",
                "use a captured operation or a typed qfunc; host calls are not executed",
            )

    def region_marker(self, block: ast.With) -> object:
        if len(block.items) == 1:
            item = block.items[0]
            expression = item.context_expr
            if (
                item.optional_vars is None
                and isinstance(expression, ast.Call)
                and not expression.args
                and not expression.keywords
            ):
                return _resolve(self.definition, expression.func)
        return None

    def declare_local(self, node: ast.AnnAssign) -> str:
        if not isinstance(node.target, ast.Name) or node.value is None:
            _error(
                self.source,
                node,
                "program.local",
                "assign an arithmetic expression to a fresh sized QNum local",
            )
        name = node.target.id
        if name in self.ports:
            _error(
                self.source,
                node,
                "program.local",
                "a local cannot shadow a quantum name",
            )
        annotation = _annotation_node(self.source, node.annotation)
        if (
            not isinstance(annotation, ast.Subscript)
            or _resolve(self.definition, annotation.value, annotation=True) is not QNum
            or not isinstance(annotation.slice, ast.Constant)
        ):
            _error(
                self.source,
                node,
                "program.local",
                "local declaration requires QNum[literal width]",
            )
        try:
            width = _width_spec(annotation.slice.value)
        except (TypeError, ValueError) as error:
            _error(self.source, node, "program.local", str(error))
        value = Value(ValueId(self.name, self.counter), register_type(width))
        self.counter += 1
        self.ports[name] = _Port(name, "output", "qnum", width)
        self.widths[name] = width
        self.registers[name] = value
        self.operations.append(
            Operation(
                "qreg.local",
                results=(value,),
                attributes={"name": f"{name}.{value.id.index}"},
                location=self.source.location(node),
            )
        )
        graph, actuals = self.capture_expression(node.value, node)
        self.emit("qreg.compute", node, (value, *actuals), callee=graph.name)
        return name

    def within_apply(self, prepare: ast.With, action: ast.stmt | None) -> None:
        if (
            self.region_marker(prepare) is not within
            or not isinstance(action, ast.With)
            or self.region_marker(action) is not apply
        ):
            _error(
                self.source,
                prepare,
                "program.region",
                "pair with within() with a following with apply()",
            )
        locals_in_order: list[str] = []
        for statement in prepare.body:
            if isinstance(statement, ast.AnnAssign):
                locals_in_order.append(self.declare_local(statement))
            else:
                _error(
                    self.source,
                    statement,
                    "program.within",
                    "within prepares fresh arithmetic locals only",
                )
        if not locals_in_order:
            _error(
                self.source,
                prepare,
                "program.within",
                "within requires a local computation",
            )
        for statement in action.body:
            if (
                isinstance(statement, ast.With)
                and self.region_marker(statement) is adjoint
            ):
                self.adjoint_scope(statement)
                continue
            if not isinstance(statement, ast.Expr) or not isinstance(
                statement.value, ast.Call
            ):
                _error(
                    self.source,
                    statement,
                    "program.apply",
                    "apply accepts phase and Const-only qfunc calls",
                )
            function = _resolve(self.definition, statement.value.func)
            if function is not phase and (
                not isinstance(function, _QuantumFunction)
                or not function.source.program
                or any(p.role != "const" for p in program_ports(function))
            ):
                _error(
                    self.source,
                    statement,
                    "program.apply",
                    "apply accepts phase and Const-only qfunc calls",
                )
            self.call(statement.value)
        for name in reversed(locals_in_order):
            value = self.registers.pop(name)
            self.emit("qreg.uncompute", prepare, (value,))
            self.emit("qreg.release", prepare, (value,))
            del self.ports[name]
            del self.widths[name]

    def adjoint_scope(self, block: ast.With) -> None:
        for statement in reversed(block.body):
            if not isinstance(statement, ast.Expr) or not isinstance(
                statement.value, ast.Call
            ):
                _error(
                    self.source,
                    statement,
                    "program.adjoint",
                    "adjoint requires static qfunc calls",
                )
            self.call(statement.value, inverse=True)

    def finish(self) -> HybridProgram:
        statements = iter(self.source.node.body)
        for statement in statements:
            if isinstance(statement, ast.AugAssign):
                self.assign(statement)
            elif isinstance(statement, ast.With):
                if self.region_marker(statement) is adjoint:
                    self.adjoint_scope(statement)
                else:
                    self.within_apply(statement, next(statements, None))
            elif isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Call
            ):
                self.call(statement.value)
        values = {
            name: self.register(
                ast.copy_location(ast.Name(id=name, ctx=ast.Load()), self.source.node)
            )
            for name in self.ports
        }
        borrowed = tuple(
            values[p.name] for p in self.ports.values() if p.role == "const"
        )
        outputs = tuple(
            values[p.name] for p in self.ports.values() if p.role == "output"
        )
        declarations = tuple(
            Operation(
                "qreg.declare",
                results=(values[p.name],),
                attributes={"name": p.name},
                location=self.source.location(self.source.node),
            )
            for p in self.ports.values()
            if p.role == "output"
        )
        return HybridProgram(
            self.name,
            Region(
                (
                    Block(
                        (self.entry, *borrowed),
                        (
                            *declarations,
                            *self.operations,
                            Operation(
                                "qreg.return",
                                (self.effect, *borrowed, *outputs),
                                location=self.source.location(self.source.node),
                            ),
                        ),
                    ),
                )
            ),
            location=self.source.location(self.source.node),
        )


def capture_program(
    definition: _QuantumFunction,
) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    """Capture the reachable typed graph; existing Compiler verifies all effects."""
    owner = _Capture(definition)
    program, _ = owner.build(definition, root=True)
    functions = tuple(owner.functions)
    verify_register_program(program, functions)
    return program, functions
