"""Test-only producers of the shared register profile; no native source parser."""

from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_profile import register_type
from flagquantum.compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)


class Registers:
    """Keep test records readable while preserving explicit effect operands."""

    def __init__(self) -> None:
        self.entry = Value(ValueId("register_probe", 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.outputs: list[Value] = []
        self.operations: list[Operation] = []
        self.counter = 1

    def declare(self, name: str, width: int) -> Value:
        register = Value(ValueId("register_probe", self.counter), register_type(width))
        self.counter += 1
        self.outputs.append(register)
        self.operations.append(
            Operation("qreg.declare", results=(register,), attributes={"name": name})
        )
        return register

    def emit(self, name: str, *registers: Value, **attributes: object) -> None:
        effect = Value(ValueId("register_probe", self.counter), QUANTUM_EFFECT)
        self.counter += 1
        self.operations.append(
            Operation(
                name,
                (self.effect, *registers),
                (effect,),
                attributes,
                location=SourceLocation(
                    "register_probe.py", len(self.operations) + 1, 1
                ),
            )
        )
        self.effect = effect

    def finish(self) -> HybridProgram:
        return HybridProgram(
            "register_probe",
            Region(
                (
                    Block(
                        (self.entry,),
                        (
                            *self.operations,
                            Operation("qreg.return", (self.effect, *self.outputs)),
                        ),
                    ),
                )
            ),
        )


def square_plus_one(x: int) -> int:
    return x * x + 1


def product(a: int, b: int) -> int:
    return a * b


def double(x: int) -> int:
    return x + x


def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u


def pressure_program() -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    graph = capture(pressure, {"x": 1, "y": 1})
    p = Registers()
    x, y, z = p.declare("x", 1), p.declare("y", 1), p.declare("result", 2)
    for value in (x, y):
        p.emit("qreg.allocate", value)
        p.emit("qreg.gate", value, gate="h", index=0)
    p.emit("qreg.gate", x, gate="s", index=0)
    p.emit("qreg.compute", z, x, y, callee=graph.name)
    return p.finish(), (graph,)


def tutorial_program(width: int = 2) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    graph = capture(square_plus_one, {"x": width})
    out_width = graph.body.blocks[0].operations[-1].operands[0].type.parameters[0]
    p = Registers()
    x, y = p.declare("x", width), p.declare("y", out_width)
    p.emit("qreg.allocate", x)
    for index in range(width):
        p.emit("qreg.gate", x, gate="h", index=index)
    p.emit("qreg.compute", y, x, callee=graph.name)
    return p.finish(), (graph,)
