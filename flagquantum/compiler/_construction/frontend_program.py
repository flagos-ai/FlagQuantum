"""Capture typed Python programs into existing checked register/effect records."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace

from .._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from .frontend_classical import ClassicalValue, bind_inputs, real_number
from .frontend_evolution import _Evolution, capture_suzuki_trotter, flush_evolutions
from .frontend_matrix import capture_unitary
from .frontend_phase import capture_polynomial_phase
from .frontend_ports import (
    _annotation_node,
    _Port,
    _resolve,
    classical_parameters,
    program_ports,
)
from .frontend_preparation import capture_preparation
from .frontend_python import _error, _QuantumFunction, _Source, capture
from .frontend_statements import capture_statements
from .frontend_views import gate_target, inline_mutable, locations, static_integer
from .profile import _width
from .program_types import (
    QNum,
    _width_spec,
    adjoint,
    allocate,
    apply,
    control,
    crx,
    cx,
    h,
    hadamard_transform,
    modular_exp,
    phase,
    phase_gate,
    prepare_amplitudes,
    prepare_bell_state,
    prepare_ghz_state,
    prepare_state,
    prepare_uniform_interval_state,
    qft,
    r,
    rx,
    rz,
    rzz,
    s,
    suzuki_trotter,
    swap,
    u,
    unitary,
    within,
    x,
)
from .ranges import unsigned_width
from .register_profile import (
    equatorial_rotation,
    euler_rotation,
    register_type,
    register_width,
    verify_register_program,
)


class _Capture:
    def __init__(
        self, entry: _QuantumFunction, classical: Mapping[str, ClassicalValue]
    ) -> None:
        self.classical = classical
        self.entry = entry.source.node.name
        self.functions: list[HybridProgram] = []
        self.active: set[int] = set()
        self.seen: set[int] = set()
        self.nodes = 0
        self.serial = 0
        self.inline_calls = 0
        self.loop_steps = 0
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
            builder = _Body(
                self, definition, name, ports, self.classical if root else None
            )
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
        classical: Mapping[str, ClassicalValue] | None = None,
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
        self.evolutions: list[_Evolution] = []
        self.branch_depth = 0
        self.expressions = 0
        self.aliases: dict[str, tuple[tuple[Value, int], ...]] | None = None
        self.array_names = {p.name for p in ports if p.kind == "qarray"}
        self.loop_variables: dict[str, int] = {}
        self.classical = dict(classical or {})

    def register(self, node: ast.expr, inferred: int | None = None) -> Value:
        if self.aliases is not None:
            _error(
                self.source,
                node,
                "program.borrow",
                "mutable helpers accept gate effects only",
            )
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
        flush_evolutions(self)
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
        if not isinstance(node.op, ast.BitOr):
            _error(
                self.source,
                node,
                "program.expression",
                "initialize fresh outputs with |= only",
            )
        if (
            isinstance(node.target, ast.Name)
            and node.target.id in self.ports
            and self.ports[node.target.id].kind == "qarray"
        ):
            _error(
                self.source,
                node,
                "program.expression",
                "initialize arrays with allocate or a preparation, not numeric assignment",
            )
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
        allowed: tuple[type[ast.AST], ...] = (
            ast.BinOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Mult,
            ast.Pow,
            ast.Mod,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.UnaryOp,
            ast.Invert,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.IfExp,
        )
        if any(not isinstance(child, allowed) for child in ast.walk(expr)):
            _error(
                self.source,
                expr,
                "program.expression",
                "use unsigned arithmetic, comparisons, conditional values, bitwise values or an unsigned literal",
            )
        used = {child.id for child in ast.walk(expr) if isinstance(child, ast.Name)}
        if not used or not used.issubset(self.ports):
            _error(
                self.source,
                expr,
                "program.expression",
                "arithmetic reads declared quantum ports; basis initialization uses an unsigned literal",
            )
        if any(self.ports[name].kind == "qarray" for name in used):
            _error(
                self.source,
                expr,
                "program.expression",
                "array arithmetic is outside the numeric profile",
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
        return real_number(self, node)

    def resolve_callable(self, node: ast.expr) -> object:
        """Resolve operation identities while respecting all captured local scopes."""
        callable_root = node
        while isinstance(callable_root, ast.Attribute):
            callable_root = callable_root.value
        if isinstance(callable_root, ast.Name) and (
            callable_root.id
            in (self.aliases if self.aliases is not None else self.ports)
            or callable_root.id in self.classical
            or callable_root.id in self.loop_variables
        ):
            _error(
                self.source,
                node,
                "program.call",
                "a quantum port cannot shadow a callable",
            )
        return _resolve(self.definition, node)

    def call(
        self,
        node: ast.Call,
        *,
        inverse: bool = False,
        control_bit: Value | None = None,
        condition_inputs: tuple[Value, ...] = (),
    ) -> None:
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            _error(
                self.source,
                node,
                "program.signature",
                "use positional arguments without unpacking",
            )
        function = self.resolve_callable(node.func)
        if function is not suzuki_trotter:
            flush_evolutions(self)
        if self.aliases is not None and not (
            any(
                function is marker
                for marker in (
                    h,
                    s,
                    x,
                    rx,
                    rz,
                    phase_gate,
                    phase,
                    control,
                    rzz,
                    cx,
                    swap,
                    crx,
                    r,
                    u,
                    hadamard_transform,
                    qft,
                    unitary,
                    modular_exp,
                    suzuki_trotter,
                )
            )
            or isinstance(function, _QuantumFunction)
            and function.source.program
        ):
            _error(
                self.source,
                node,
                "program.borrow",
                "mutable helpers allow gates and mutable helper calls only",
            )
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
        if function is control:
            self.scalar_control(node)
        elif function is allocate:
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
        elif function is unitary:
            capture_unitary(self, node)
        elif function is suzuki_trotter:
            capture_suzuki_trotter(self, node)
        elif function is modular_exp:
            if len(node.args) != 4:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "modular_exp takes modulus, base, target and exponent",
                )
            modulus, base = (static_integer(self, arg) for arg in node.args[:2])
            modular_operands = []
            for argument in node.args[2:]:
                if (
                    not isinstance(argument, ast.Name)
                    or argument.id not in self.ports
                    or self.ports[argument.id].kind not in {"qnum", "qbit"}
                ):
                    _error(
                        self.source,
                        argument,
                        "program.modular_exp",
                        "modular_exp requires complete numeric registers",
                    )
                targets = locations(self, argument)
                destination = targets[0][0]
                if targets != tuple(
                    (destination, bit) for bit in range(register_width(destination))
                ):
                    _error(
                        self.source,
                        argument,
                        "program.view",
                        "modular_exp requires complete registers in logical bit order",
                    )
                modular_operands.append(destination)
            self.emit(
                "qreg.modular_exp",
                node,
                tuple(modular_operands),
                modulus=modulus,
                base=base,
            )
        elif function is qft:
            if len(node.args) != 1:
                _error(self.source, node, "program.signature", "qft takes one register")
            targets = locations(self, node.args[0])
            destination = targets[0][0]
            if targets != tuple(
                (destination, bit) for bit in range(register_width(destination))
            ):
                _error(
                    self.source,
                    node,
                    "program.view",
                    "qft requires a complete register in logical bit order",
                )
            self.emit("qreg.qft", node, (destination,))
        elif function is hadamard_transform:
            if len(node.args) != 1:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "hadamard_transform takes one quantum port",
                )
            for destination, bit in locations(self, node.args[0]):
                self.emit("qreg.gate", node, (destination,), gate="h", index=bit)
        elif any(function is gate for gate in (x, h, s, rx, rz, phase_gate, rzz)):
            angled = not any(function is gate for gate in (x, h, s))
            if len(node.args) != 1 + angled:
                _error(self.source, node, "program.signature", "invalid gate arity")
            attributes: dict[str, object] = {}
            if angled:
                angle = self.phase_number(node.args[0])
                if angle is None:
                    _error(
                        self.source,
                        node,
                        "program.angle",
                        "gate angle must be a finite static real",
                    )
                attributes["angle"] = angle
            if function is rzz:
                targets = locations(self, node.args[-1])
                if len(targets) != 2:
                    _error(
                        self.source,
                        node,
                        "program.width",
                        "RZZ requires exactly two bits",
                    )
                values = tuple(value for value, _ in targets)
                indices = tuple(index for _, index in targets)
                self.emit(
                    "qreg.pair_gate",
                    node,
                    values,
                    gate="rzz",
                    indices=indices,
                    **attributes,
                )
            else:
                destination, index = gate_target(self, node.args[-1])
                gate = next(
                    name
                    for marker, name in (
                        (x, "x"),
                        (h, "h"),
                        (s, "s"),
                        (rx, "rx"),
                        (rz, "rz"),
                        (phase_gate, "phase"),
                    )
                    if function is marker
                )
                self.emit(
                    "qreg.gate",
                    node,
                    (destination,),
                    gate=gate,
                    index=index,
                    **attributes,
                )
        elif any(
            function is preparation
            for preparation in (
                prepare_bell_state,
                prepare_ghz_state,
                prepare_amplitudes,
                prepare_state,
            )
        ):
            capture_preparation(self, node, function)
        elif function is cx or function is swap or function is crx:
            angled = function is crx
            if len(node.args) != 2 + angled:
                _error(
                    self.source, node, "program.signature", "invalid pair gate arity"
                )
            attributes = {"gate": function.__name__}
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
            targets = tuple(gate_target(self, arg) for arg in node.args[int(angled) :])
            operands = tuple(value for value, _ in targets)
            indices = tuple(index for _, index in targets)
            if any(register_width(value) != 1 for value in operands):
                attributes["indices"] = indices
            self.emit("qreg.pair_gate", node, operands, **attributes)
        elif function is prepare_uniform_interval_state:
            if len(node.args) != 3:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "expected two bounds and a target",
                )
            start, stop, output = node.args
            if (
                not isinstance(start, ast.Constant)
                or type(start.value) is not int
                or not isinstance(stop, ast.Constant)
                or type(stop.value) is not int
            ):
                _error(
                    self.source,
                    node,
                    "program.prepare",
                    "uniform preparation requires literal integer bounds",
                )
            destination = self.register(output)
            self.emit(
                "qreg.uniform_interval",
                node,
                (destination,),
                start=start.value,
                stop=stop.value,
            )
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
            destination, index = gate_target(self, node.args[2])
            for gate, angle in equatorial_rotation(theta, phi):
                self.emit(
                    "qreg.gate",
                    node,
                    (destination,),
                    gate=gate,
                    index=index,
                    angle=angle,
                )
        elif function is u:
            if len(node.args) != 5:
                _error(
                    self.source,
                    node,
                    "program.signature",
                    "U requires four angles and a target",
                )
            theta, phi, lam, gamma = (self.phase_number(arg) for arg in node.args[:4])
            if theta is None or phi is None or lam is None or gamma is None:
                _error(
                    self.source,
                    node,
                    "program.angle",
                    "U requires finite static real angles",
                )
            destination, index = gate_target(self, node.args[4])
            for gate, angle in euler_rotation(theta, phi, lam):
                self.emit(
                    "qreg.gate",
                    node,
                    (destination,),
                    gate=gate,
                    index=index,
                    angle=angle,
                )
            self.emit("qreg.phase", node, (), angle=gamma)
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
            elif self.aliases is not None or any(
                isinstance(child, (ast.Subscript, ast.Sub, ast.USub, ast.Div))
                or isinstance(child, ast.Name)
                and child.id in self.classical
                or isinstance(child, ast.Constant)
                and type(child.value) is float
                for child in ast.walk(expr)
            ):
                capture_polynomial_phase(self, expr, angle, node)
            else:
                graph, actuals = self.capture_expression(expr, node)
                width = _width(
                    graph.body.blocks[0].operations[-1].operands[0].type, 256
                )
                attributes = {"angle": angle} if angle != math.pi or width != 1 else {}
                self.emit("qreg.phase", node, actuals, callee=graph.name, **attributes)
        elif isinstance(function, _QuantumFunction) and function.source.program:
            formal = program_ports(function)
            if any(port.role == "inout" for port in formal):
                if inverse or control_bit is not None:
                    _error(
                        self.source,
                        node,
                        "program.borrow",
                        "mutable helpers are not admitted under adjoint/control",
                    )
                inline_mutable(self, node, function, formal)
                return
            if classical_parameters(function):
                _error(
                    self.source,
                    node,
                    "program.binding",
                    "parameterized helpers require plain writable quantum ports",
                )
            if self.aliases is not None:
                _error(
                    self.source,
                    node,
                    "program.borrow",
                    "mutable helper calls cannot initialize or publish Outputs",
                )
            if control_bit is not None and any(port.role != "const" for port in formal):
                _error(
                    self.source,
                    node,
                    "program.control",
                    "controlled helpers require Const-only ports",
                )
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
            if control_bit is None:
                self.emit("qreg.call", node, (*reads, *writes), **call_attributes)
            else:
                if len(set(reads)) != len(reads) or set(reads).intersection(
                    condition_inputs
                ):
                    _error(
                        self.source,
                        node,
                        "program.control",
                        "controlled helper arguments must be disjoint",
                    )
                self.emit(
                    "qreg.control_call", node, (control_bit, *reads), **call_attributes
                )
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
                return self.resolve_callable(expression.func)
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
        if name in self.ports or name in self.classical or name in self.loop_variables:
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
            function = self.resolve_callable(statement.value.func)
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

    def scalar_control(self, node: ast.Call) -> None:
        if len(node.args) != 2 or not isinstance(node.args[1], ast.Lambda):
            _error(
                self.source,
                node,
                "program.control",
                "expected control(bit, lambda: operation)",
            )
        condition, callback = node.args
        assert isinstance(callback, ast.Lambda)
        args = callback.args
        body = callback.body
        if (
            args.posonlyargs
            or args.args
            or args.kwonlyargs
            or args.vararg
            or args.kwarg
            or not isinstance(body, ast.Call)
            or body.keywords
            or any(isinstance(arg, ast.Starred) for arg in body.args)
        ):
            _error(
                self.source,
                callback,
                "program.control",
                "expected one static operation in a zero-argument lambda",
            )
        assert isinstance(body, ast.Call)
        if isinstance(body.func, ast.Name) and body.func.id in self.ports:
            _error(
                self.source,
                body.func,
                "program.control",
                "a quantum port cannot shadow a callable",
            )
        operation = self.resolve_callable(body.func)
        if self.aliases is not None and operation is not phase:
            _error(
                self.source,
                body,
                "program.borrow",
                "mutable helpers allow phase-only control",
            )
        phase_constant = None
        if operation is phase and len(body.args) == 2:
            phase_constant = self.phase_number(body.args[0])
        if (
            operation is phase
            and len(body.args) == 2
            and (
                self.aliases is not None
                or isinstance(condition, ast.Subscript)
                or phase_constant is None
                and (
                    not isinstance(body.args[0], ast.Name)
                    or any(
                        child.id == body.args[0].id
                        for child in ast.walk(condition)
                        if isinstance(child, ast.Name)
                    )
                )
            )
        ):
            expression, coefficient = body.args
            angle = self.phase_number(coefficient)
            if angle is None:
                _error(
                    self.source,
                    coefficient,
                    "program.control",
                    "expected a finite static phase coefficient",
                )
            capture_polynomial_phase(self, expression, angle, body, condition=condition)
            return
        if isinstance(condition, ast.Compare):
            graph, inputs = self.capture_expression(condition, node)
            if sum(register_width(value) for value in inputs) > 8:
                _error(
                    self.source,
                    condition,
                    "program.control",
                    "control predicate exceeds eight input bits",
                )
            predicate = Value(ValueId(self.name, self.counter), register_type(1))
            self.counter += 1
            self.operations.append(
                Operation(
                    "qreg.local",
                    results=(predicate,),
                    attributes={"name": f"control.{predicate.id.index}"},
                    location=self.source.location(node),
                )
            )
            self.emit("qreg.compute", node, (predicate, *inputs), callee=graph.name)
        else:
            predicate = self.register(condition)
            inputs = (predicate,)
        if operation is hadamard_transform and len(body.args) == 1:
            target = self.register(body.args[0])
            if target in inputs:
                _error(
                    self.source,
                    node,
                    "program.control",
                    "condition and target must be disjoint",
                )
            self.emit("qreg.control_h", node, (predicate, target))
        elif operation is phase and len(body.args) == 2:
            expression, coefficient = body.args
            angle = self.phase_number(coefficient)
            if angle is None:
                _error(
                    self.source,
                    coefficient,
                    "program.control",
                    "expected a finite static phase coefficient",
                )
            constant = phase_constant
            operands: tuple[Value, ...]
            if constant is None:
                target = self.register(expression)
                if target in inputs:
                    _error(
                        self.source,
                        node,
                        "program.control",
                        "condition and target must be disjoint",
                    )
                operands = (predicate, target)
            else:
                operands = (predicate,)
                angle *= constant
                if not math.isfinite(angle):
                    _error(
                        self.source,
                        expression,
                        "program.control",
                        "controlled phase angle exceeds finite range",
                    )
            self.emit("qreg.control_phase", node, operands, angle=angle)
        elif isinstance(operation, _QuantumFunction) and operation.source.program:
            self.call(body, control_bit=predicate, condition_inputs=inputs)
        else:
            _error(
                self.source,
                body,
                "program.control",
                "expected scalar phase or Hadamard",
            )

        if isinstance(condition, ast.Compare):
            self.emit("qreg.uncompute", node, (predicate,))
            self.emit("qreg.release", node, (predicate,))

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

    def statements(self, body: Sequence[ast.stmt], *, nested: bool = False) -> None:
        capture_statements(self, body, nested=nested)

    def finish(self) -> HybridProgram:
        self.statements(self.source.node.body)
        flush_evolutions(self)
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
    classical_inputs: Mapping[str, float | Sequence[float]] | None = None,
) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    """Capture the reachable typed graph; existing Compiler verifies all effects."""
    owner = _Capture(definition, bind_inputs(definition, classical_inputs))
    program, _ = owner.build(definition, root=True)
    functions = tuple(owner.functions)
    verify_register_program(program, functions)
    return program, functions
