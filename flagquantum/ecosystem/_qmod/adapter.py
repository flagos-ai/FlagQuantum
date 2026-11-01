"""Native syntax to the admitted Compiler models, without synthesis or execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...compiler._construction.profile import uint_type
from ...compiler._construction.ranges import unsigned_width
from ...compiler._construction.register_profile import (
    register_type,
    verify_register_program,
)
from ...compiler._hybrid.model import (
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    Value,
    ValueId,
)
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from .syntax import (
    Expression,
    Function,
    Parameter,
    Parser,
    Statement,
    Token,
    reject,
    tokenize,
)

SUPPORTED_SOURCE_VERSION = "2bfd835e73b6882fcc515a725098b04766c238e7"
MAX_EFFECT_OPERATIONS = 4096
_GATE_CALLS = frozenset({"X", "H", "S", "Z", "hadamard_transform"})
_INTRINSICS = _GATE_CALLS | {"allocate", "phase", "pi"}
_RELATIONS = MappingProxyType(
    {
        "==": ("qint.equal", False, False),
        "!=": ("qint.equal", False, True),
        "<": ("qint.less", False, False),
        "<=": ("qint.less", True, True),
        ">": ("qint.less", True, False),
        ">=": ("qint.less", False, True),
    }
)


@dataclass(frozen=True)
class QmodImport:
    """Private construction handoff; source_version is declared, not authenticated."""

    program: HybridProgram
    functions: tuple[HybridProgram, ...]
    source_version: str
    inferred_widths: Mapping[str, int]
    source_name: str


class _Expression:
    """Infer unsigned arithmetic/predicate domains and emit existing qint records."""

    def __init__(
        self, node: Expression, widths: Mapping[str, int | None], name: str
    ) -> None:
        self.node, self.widths, self.name = node, widths, name
        self.facts: dict[Expression, tuple[int, bool]] = {}
        self.names: list[str] = []
        self.comparison = False
        self.bound, self.quantum = self.infer(node)
        arguments: list[Value] = []
        for index, key in enumerate(self.names):
            width = widths[key]
            assert width is not None  # infer already refuses unresolved inputs.
            arguments.append(Value(ValueId(name, index), uint_type(width)))
        self.arguments = tuple(arguments)
        self.inputs = dict(zip(self.names, self.arguments, strict=True))
        self.operations: list[Operation] = []

    def infer(self, node: Expression) -> tuple[int, bool]:
        text = node.token.text
        if not node.children:
            if text.isdecimal():
                result = (int(text), False)
            else:
                if text not in self.widths:
                    reject(node.token, "name", f"unknown output {text!r}")
                width = self.widths[text]
                if width is None:
                    reject(
                        node.token,
                        "width",
                        f"{text!r} has no initialized numeric width",
                    )
                if text not in self.names:
                    self.names.append(text)
                result = ((1 << width) - 1, True)
        else:
            left, lq = self.infer(node.children[0])
            right, rq = self.infer(node.children[1])
            if text in _RELATIONS:
                self.comparison = True
                opcode, swap, invert = _RELATIONS[text]
                a, b = (right, left) if swap else (left, right)
                bound = int(a == b if opcode == "qint.equal" else a < b) ^ int(invert)
                result = (1 if lq or rq else bound, lq or rq)
            else:
                bound = (
                    left + right
                    if text == "+"
                    else left * left if text == "**" else left * right
                )
                result = (bound, lq or rq)
        if unsigned_width(result[0]) > 256:
            reject(
                node.token,
                "width",
                "expression domain exceeds the 256-bit unsigned profile",
            )
        self.facts[node] = result
        return result

    def operation(
        self,
        name: str,
        operands: tuple[Value, ...],
        kind: IRType,
        token: Token,
        **attributes: object,
    ) -> Value:
        result = Value(
            ValueId(self.name, len(self.arguments) + len(self.operations)), kind
        )
        self.operations.append(
            Operation(name, operands, (result,), attributes, location=token.location)
        )
        return result

    def emit(self, node: Expression) -> Value:
        bound, quantum = self.facts[node]
        if not quantum:
            return self.operation("arith.constant", (), INDEX, node.token, value=bound)
        if not node.children:
            return self.inputs[node.token.text]
        left = self.emit(node.children[0])
        right = left if node.token.text == "**" else self.emit(node.children[1])
        if node.token.text not in _RELATIONS:
            return self.operation(
                "qint.add" if node.token.text == "+" else "qint.multiply",
                (left, right),
                uint_type(unsigned_width(bound)),
                node.token,
            )
        opcode, swap, invert = _RELATIONS[node.token.text]
        result = self.operation(
            opcode, (right, left) if swap else (left, right), uint_type(1), node.token
        )
        if invert:
            zero = self.operation("arith.constant", (), INDEX, node.token, value=0)
            result = self.operation(
                "qint.equal", (result, zero), uint_type(1), node.token
            )
        return result

    def function(self) -> HybridProgram:
        returned = self.emit(self.node)
        return HybridProgram(
            self.name,
            Region(
                (
                    Block(
                        self.arguments,
                        (
                            *self.operations,
                            Operation(
                                "qint.return",
                                (returned,),
                                location=self.node.token.location,
                            ),
                        ),
                    ),
                )
            ),
            location=self.node.token.location,
        )


class _Adapt:
    def __init__(
        self,
        syntax: Function,
        resolve: Callable[[Token], tuple[Function, HybridProgram]],
    ) -> None:
        self.syntax = syntax
        self.resolve = resolve
        self.scope = syntax.name.text
        self.parameters: dict[str, Parameter] = {}
        for parameter in syntax.parameters:
            name = parameter.name.text
            if name in self.parameters:
                reject(parameter.name, "name", f"duplicate output {name!r}")
            self.parameters[name] = parameter
        self.widths = {name: p.width for name, p in self.parameters.items()}
        self.inferred: dict[str, int] = {}
        self.registers: dict[str, Value] = {}
        self.counter = len(self.parameters) + 1
        self.entry = Value(ValueId(self.scope, 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.operations: list[Operation] = []
        self.functions: list[HybridProgram] = []
        self.local_scope: list[tuple[str, Token]] | None = None

    def register(self, token: Token, inferred: int | None = None) -> Value:
        name = token.text
        if name not in self.parameters:
            reject(token, "name", f"unknown output {name!r}")
        local = self.parameters[name].role == "local"
        if local and self.local_scope is None:
            reject(
                token,
                "scope",
                "local references are available only inside within/apply",
            )
        width = self.widths[name]
        if width is None and inferred is not None:
            self.widths[name] = width = inferred
            if not local:
                self.inferred[name] = inferred
        if width is None:
            reject(
                token,
                "width",
                f"{name!r} needs a declared width or explicit initialization size",
            )
        if inferred is not None and inferred > width:
            reject(
                token,
                "width",
                f"declared width {width} cannot fit the expression domain requiring {inferred} bits",
            )
        if name not in self.registers:
            if local:
                index = self.counter
                self.counter += 1
            else:
                index = tuple(self.parameters).index(name) + 1
            self.registers[name] = Value(
                ValueId(self.scope, index), register_type(width)
            )
            if local:
                assert self.local_scope is not None
                self.local_scope.append((name, token))
                self.append_op(
                    token,
                    Operation(
                        "qreg.local",
                        results=(self.registers[name],),
                        attributes={"name": f"{name}.within{index}"},
                        location=token.location,
                    ),
                )
        return self.registers[name]

    def append_op(self, token: Token, operation: Operation) -> None:
        if len(self.operations) >= MAX_EFFECT_OPERATIONS:
            reject(
                token,
                "budget.expansion",
                "native expansion exceeds 4096 effect operations",
            )
        self.operations.append(operation)

    def effect_op(
        self, name: str, token: Token, values: tuple[Value, ...], **attributes: object
    ) -> None:
        effect = Value(ValueId(self.scope, self.counter), QUANTUM_EFFECT)
        self.counter += 1
        self.append_op(
            token,
            Operation(
                name,
                (self.effect, *values),
                (effect,),
                attributes,
                location=token.location,
            ),
        )
        self.effect = effect

    @staticmethod
    def atom(node: Expression) -> Token:
        if node.children or node.token.text.isdecimal():
            reject(node.token, "signature", "this argument must be one output variable")
        return node.token

    def call(
        self,
        token: Token,
        arguments: tuple[Expression, ...],
        *,
        const_only: bool = False,
    ) -> None:
        if token.text in self.parameters:
            reject(token, "call", "an output variable shadows this helper call")
        syntax, program = self.resolve(token)
        if const_only and any(p.role != "const" for p in syntax.parameters):
            reject(token, "scope", "apply helper calls require Const-only parameters")
        if len(arguments) != len(syntax.parameters):
            reject(token, "signature", "helper argument count disagrees")
        const = iter(program.body.blocks[0].arguments[1:])
        outputs = iter(
            op.results[0]
            for op in program.body.blocks[0].operations
            if op.name == "qreg.declare"
        )
        reads: list[Value] = []
        writes: list[Value] = []
        for argument, parameter in zip(arguments, syntax.parameters, strict=True):
            borrowed = parameter.role == "const"
            formal = next(const if borrowed else outputs)
            actual_name = self.atom(argument)
            actual = self.register(
                actual_name, None if borrowed else formal.type.parameters[0]
            )
            if (
                self.parameters[actual_name.text].kind != parameter.kind
                or actual.type != formal.type
            ):
                reject(
                    argument.token,
                    "signature",
                    "helper argument kind/width must match exactly",
                )
            (reads if borrowed else writes).append(actual)
        self.effect_op("qreg.call", token, (*reads, *writes), callee=program.name)

    def within(self, statement: Statement, index: str) -> None:
        if not statement.within:
            reject(
                statement.token,
                "scope",
                "within requires at least one local computation",
            )
        self.local_scope = []
        try:
            for position, child in enumerate(statement.within):
                if child.token.text != "=":
                    reject(
                        child.token,
                        "scope",
                        "within admits only quantum-expression local assignments",
                    )
                target = self.atom(child.arguments[0])
                if (
                    target.text not in self.parameters
                    or self.parameters[target.text].role != "local"
                ):
                    reject(target, "scope", "within prepares declared locals only")
                self.statement(child, f"{index}.within{position}")
            for position, child in enumerate(statement.apply):
                if child.token.text == "phase":
                    self.statement(child, f"{index}.apply{position}")
                elif child.token.text in _INTRINSICS or child.token.text == "=":
                    reject(
                        child.token,
                        "scope",
                        "apply admits phase and Const-only helper calls",
                    )
                else:
                    self.call(child.token, child.arguments, const_only=True)
            for name, token in reversed(self.local_scope):
                value = self.registers[name]
                self.effect_op("qreg.uncompute", token, (value,))
                self.effect_op("qreg.release", token, (value,))
        finally:
            for name, _ in self.local_scope:
                self.registers.pop(name, None)
            self.local_scope = None

    def statement(self, statement: Statement, index: str) -> None:
        token, args = statement.token, statement.arguments
        if statement.declaration is not None:
            parameter = statement.declaration
            name = parameter.name.text
            if name in self.parameters or name in _INTRINSICS:
                reject(
                    parameter.name,
                    "name",
                    "local names cannot shadow parameters, locals or intrinsics",
                )
            self.parameters[name] = parameter
            self.widths[name] = parameter.width
        elif statement.within is not None:
            self.within(statement, index)
        elif token.text == "=":
            target = self.atom(args[0])
            expression = _Expression(args[1], self.widths, f"{self.scope}.expr{index}")
            if self.local_scope is not None and not expression.quantum:
                reject(
                    token,
                    "scope",
                    "within preparation requires a quantum-dependent expression",
                )
            destination = self.register(target, unsigned_width(expression.bound))
            if expression.quantum:
                graph = expression.function()
                self.functions.append(graph)
                inputs = tuple(
                    self.register(self.parameters[name].name)
                    for name in expression.names
                )
                self.effect_op(
                    "qreg.compute", token, (destination, *inputs), callee=graph.name
                )
            else:
                self.effect_op(
                    "qreg.initialize", token, (destination,), value=expression.bound
                )
        elif token.text == "phase":
            if len(args) != 2 or "phase" in self.parameters:
                reject(
                    token,
                    "signature",
                    "phase requires an expression and unshadowed pi coefficient",
                )
            coefficient = args[1]
            if (
                coefficient.children
                or coefficient.token.text != "pi"
                or "pi" in self.parameters
            ):
                reject(
                    coefficient.token,
                    "coefficient",
                    "phase supports only the unshadowed symbolic coefficient pi",
                )
            expression = _Expression(args[0], self.widths, f"{self.scope}.phase{index}")
            if expression.comparison:
                reject(
                    args[0].token,
                    "phase",
                    "phase requires a polynomial; compute comparison into a one-bit register first",
                )
            if not expression.quantum or unsigned_width(expression.bound) != 1:
                reject(
                    args[0].token,
                    "phase",
                    "phase requires a quantum expression with exactly one-bit result",
                )
            graph = expression.function()
            self.functions.append(graph)
            inputs = tuple(
                self.register(self.parameters[name].name) for name in expression.names
            )
            self.effect_op("qreg.phase", token, inputs, callee=graph.name)
        elif token.text == "allocate":
            if len(args) not in {1, 2}:
                reject(
                    token,
                    "signature",
                    "allocate accepts an output or a literal size and output",
                )
            size = None
            if len(args) == 2:
                if args[0].children or not args[0].token.text.isdecimal():
                    reject(
                        args[0].token,
                        "width",
                        "allocation size must be a literal in 1..256",
                    )
                size = int(args[0].token.text)
                if not 1 <= size <= 256:
                    reject(
                        args[0].token,
                        "width",
                        "allocation size must be a literal in 1..256",
                    )
            target = self.atom(args[-1])
            destination = self.register(target, size)
            if size is not None and self.widths[target.text] != size:
                reject(
                    target,
                    "width",
                    "allocation size must agree with the declared width",
                )
            self.effect_op("qreg.allocate", token, (destination,))
        elif token.text in _GATE_CALLS:
            if len(args) != 1:
                reject(token, "signature", f"{token.text} requires one output variable")
            target = self.atom(args[0])
            destination = self.register(target)
            if (
                token.text != "hadamard_transform"
                and self.parameters[target.text].kind != "qbit"
            ):
                reject(
                    target,
                    "type",
                    "single-qubit gates require qbit; numeric/array views are not admitted",
                )
            width = self.widths[target.text]
            assert width is not None
            for bit in range(width if token.text == "hadamard_transform" else 1):
                self.effect_op(
                    "qreg.gate",
                    token,
                    (destination,),
                    gate=(
                        "h"
                        if token.text == "hadamard_transform"
                        else token.text.lower()
                    ),
                    index=bit,
                )
        else:
            self.call(token, args)

    def adapt(self) -> HybridProgram:
        for index, statement in enumerate(self.syntax.statements):
            self.statement(statement, str(index))
        output_parameters = tuple(
            p for p in self.syntax.parameters if p.role == "output"
        )
        borrowed = tuple(
            self.register(p.name) for p in self.syntax.parameters if p.role == "const"
        )
        declared = tuple(self.register(p.name) for p in output_parameters)
        declarations = tuple(
            Operation(
                "qreg.declare",
                results=(value,),
                attributes={"name": parameter.name.text},
                location=parameter.name.location,
            )
            for parameter, value in zip(output_parameters, declared, strict=True)
        )
        program = HybridProgram(
            self.scope,
            Region(
                (
                    Block(
                        (self.entry, *borrowed),
                        (
                            *declarations,
                            *self.operations,
                            Operation(
                                "qreg.return",
                                (self.effect, *borrowed, *declared),
                                location=self.syntax.name.location,
                            ),
                        ),
                    ),
                )
            ),
            location=self.syntax.name.location,
        )
        return program


def check_source_version(source_version: str, token: Token) -> None:
    if source_version != SUPPORTED_SOURCE_VERSION:
        reject(
            token,
            "version",
            f"only the pinned source profile {SUPPORTED_SOURCE_VERSION} is admitted",
        )


def parse_qmod(
    source: str, *, filename: str = "<qmod>", source_version: str
) -> QmodImport:
    """Adapt the pinned unsigned scalar native subset to verified owned records.

    The caller declares source_version; this does not authenticate source bytes.
    Originals are independently checked against their retained source manifest.
    No files, SDKs, synthesis, execution, eval or generated Python are involved.
    """
    tokens = tokenize(source, filename)
    check_source_version(source_version, tokens[0])
    definitions: dict[str, Function] = {}
    for syntax in Parser(tokens).parse():
        name = syntax.name.text
        if name in definitions or name in _INTRINSICS:
            reject(syntax.name, "name", "duplicate function or intrinsic redefinition")
        definitions[name] = syntax
    if "main" not in definitions:
        reject(tokens[0], "entry", "source requires one qfunc main")
    compiled: dict[str, tuple[_Adapt, HybridProgram]] = {}
    active: set[str] = set()

    def resolve(token: Token) -> tuple[Function, HybridProgram]:
        name = token.text
        if name == "main" or name not in definitions:
            reject(token, "call", f"unsupported native helper call {name!r}")
        adapter, program = build(definitions[name])
        return adapter.syntax, program

    def build(syntax: Function) -> tuple[_Adapt, HybridProgram]:
        name = syntax.name.text
        if name in active:
            reject(
                syntax.name, "recursion", "quantum helper call graph must be acyclic"
            )
        if name not in compiled:
            active.add(name)
            adapter = _Adapt(syntax, resolve)
            program = adapter.adapt()
            compiled[name] = adapter, program
            active.remove(name)
        return compiled[name]

    # Unused definitions must obey the same contract as reachable helpers.
    for syntax in definitions.values():
        build(syntax)
    adapter, program = compiled["main"]
    records: dict[str, HybridProgram] = {}
    for name, (producer, model) in compiled.items():
        for function in producer.functions:
            records[function.name] = function
        if name != "main":
            records[name] = model
    functions = tuple(records.values())
    try:
        verify_register_program(program, functions)
    except HybridVerificationError as error:
        loc = program.location
        assert loc is not None
        diagnostics = tuple(
            (
                item
                if item.message.startswith(f"{loc.filename}:")
                else Diagnostic(
                    item.code, f"{loc.filename}:{loc.line}:{loc.column}: {item.message}"
                )
            )
            for item in error.diagnostics
        )
        raise HybridVerificationError(diagnostics) from error
    return QmodImport(
        program, functions, source_version, MappingProxyType(adapter.inferred), filename
    )
