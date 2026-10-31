"""Native syntax to the admitted Compiler models, without synthesis or execution."""

from __future__ import annotations

from collections.abc import Mapping
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
    Operation,
    Region,
    Value,
    ValueId,
)
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from .syntax import Expression, Function, Parameter, Parser, Token, reject, tokenize

SUPPORTED_SOURCE_VERSION = "2bfd835e73b6882fcc515a725098b04766c238e7"
MAX_EFFECT_OPERATIONS = 4096


@dataclass(frozen=True)
class QmodImport:
    """Private construction handoff; source_version is declared, not authenticated."""

    program: HybridProgram
    functions: tuple[HybridProgram, ...]
    source_version: str
    inferred_widths: Mapping[str, int]
    source_name: str


class _Expression:
    """Infer the admitted monotone unsigned domain and emit existing qint records."""

    def __init__(
        self, node: Expression, widths: Mapping[str, int | None], name: str
    ) -> None:
        self.node, self.widths, self.name = node, widths, name
        self.facts: dict[Expression, tuple[int, bool]] = {}
        self.names: list[str] = []
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

    def emit(self, node: Expression) -> Value:
        bound, quantum = self.facts[node]
        if not quantum:
            result = Value(
                ValueId(self.name, len(self.arguments) + len(self.operations)), INDEX
            )
            op = Operation(
                "arith.constant",
                results=(result,),
                attributes={"value": bound},
                location=node.token.location,
            )
        elif not node.children:
            return self.inputs[node.token.text]
        else:
            left = self.emit(node.children[0])
            right = left if node.token.text == "**" else self.emit(node.children[1])
            result = Value(
                ValueId(self.name, len(self.arguments) + len(self.operations)),
                uint_type(unsigned_width(bound)),
            )
            op = Operation(
                "qint.add" if node.token.text == "+" else "qint.multiply",
                (left, right),
                (result,),
                location=node.token.location,
            )
        self.operations.append(op)
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
    def __init__(self, syntax: Function) -> None:
        self.syntax = syntax
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
        self.entry = Value(ValueId("main", 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.operations: list[Operation] = []
        self.functions: list[HybridProgram] = []

    def register(self, token: Token, inferred: int | None = None) -> Value:
        name = token.text
        if name not in self.parameters:
            reject(token, "name", f"unknown output {name!r}")
        width = self.widths[name]
        if width is None and inferred is not None:
            self.widths[name] = width = inferred
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
            index = tuple(self.parameters).index(name) + 1
            self.registers[name] = Value(ValueId("main", index), register_type(width))
        return self.registers[name]

    def effect_op(
        self, name: str, token: Token, values: tuple[Value, ...], **attributes: object
    ) -> None:
        if len(self.operations) >= MAX_EFFECT_OPERATIONS:
            reject(
                token,
                "budget.expansion",
                "native expansion exceeds 4096 effect operations",
            )
        effect = Value(ValueId("main", self.counter), QUANTUM_EFFECT)
        self.counter += 1
        self.operations.append(
            Operation(
                name,
                (self.effect, *values),
                (effect,),
                attributes,
                location=token.location,
            )
        )
        self.effect = effect

    @staticmethod
    def atom(node: Expression) -> Token:
        if node.children or node.token.text.isdecimal():
            reject(node.token, "signature", "this argument must be one output variable")
        return node.token

    def adapt(self) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
        for index, statement in enumerate(self.syntax.statements):
            token, args = statement.token, statement.arguments
            if token.text == "=":
                target = self.atom(args[0])
                expression = _Expression(args[1], self.widths, f"main.expr{index}")
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
            elif token.text in {"X", "H", "S", "Z", "hadamard_transform"}:
                if len(args) != 1:
                    reject(
                        token, "signature", f"{token.text} requires one output variable"
                    )
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
                reject(
                    token,
                    "call",
                    f"unsupported function {token.text!r}; no operation is silently omitted",
                )
        declared = tuple(self.register(p.name) for p in self.syntax.parameters)
        declarations = tuple(
            Operation(
                "qreg.declare",
                results=(value,),
                attributes={"name": parameter.name.text},
                location=parameter.name.location,
            )
            for parameter, value in zip(self.syntax.parameters, declared, strict=True)
        )
        program = HybridProgram(
            "main",
            Region(
                (
                    Block(
                        (self.entry,),
                        (
                            *declarations,
                            *self.operations,
                            Operation(
                                "qreg.return",
                                (self.effect, *declared),
                                location=self.syntax.name.location,
                            ),
                        ),
                    ),
                )
            ),
            location=self.syntax.name.location,
        )
        functions = tuple(self.functions)
        try:
            verify_register_program(program, functions)
        except HybridVerificationError as error:
            loc = self.syntax.name.location
            diagnostics = tuple(
                (
                    item
                    if item.message.startswith(f"{loc.filename}:")
                    else Diagnostic(
                        item.code,
                        f"{loc.filename}:{loc.line}:{loc.column}: {item.message}",
                    )
                )
                for item in error.diagnostics
            )
            raise HybridVerificationError(diagnostics) from error
        return program, functions


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
    """Adapt the pinned unsigned/output native subset to verified owned records.

    The caller declares source_version; this does not authenticate source bytes.
    Originals are independently checked against their retained source manifest.
    No files, SDKs, synthesis, execution, eval or generated Python are involved.
    """
    tokens = tokenize(source, filename)
    check_source_version(source_version, tokens[0])
    adapter = _Adapt(Parser(tokens).parse())
    program, functions = adapter.adapt()
    return QmodImport(
        program, functions, source_version, MappingProxyType(adapter.inferred), filename
    )
