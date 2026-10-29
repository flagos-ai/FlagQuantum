"""Fresh register initialization and ordered effects over existing hybrid records.

Register references denote storage, while quantum.uint values remain immutable
arithmetic expressions. One linear effect orders uses; no external state is
inspected or certified by this structural verifier.
"""

from collections.abc import Sequence
from types import MappingProxyType

from .._hybrid.model import (
    QUANTUM_EFFECT,
    HybridProgram,
    IRType,
    Operation,
    Value,
    ValueId,
)
from .profile import _fail, _positive_integer, _width, verify_construction

_ATTRIBUTES = MappingProxyType(
    {
        "qreg.declare": frozenset({"name"}),
        "qreg.allocate": frozenset(),
        "qreg.initialize": frozenset({"value"}),
        "qreg.gate": frozenset({"gate", "index"}),
        "qreg.compute": frozenset({"callee"}),
        "qreg.return": frozenset(),
    }
)
_GATES = frozenset({"x", "h", "s", "z"})


def register_type(width: int) -> IRType:
    """Describe a register reference, distinct from an arithmetic value."""
    _positive_integer(width, "width")
    return IRType("quantum.register", (width,))


def register_width(value: Value, operation: Operation | None = None) -> int:
    """Check the closed register type without coercing malformed widths."""
    kind = value.type
    if (
        kind.kind != "quantum.register"
        or kind.linear
        or len(kind.parameters) != 1
        or type(kind.parameters[0]) is not int
        or kind.parameters[0] < 1
    ):
        _fail(
            "register.type",
            "expected a fixed-width quantum.register reference",
            operation,
        )
    width: int = kind.parameters[0]
    return width


def register_ports(
    program: HybridProgram,
) -> tuple[dict[str, tuple[int, ...]], dict[ValueId, tuple[int, ...]], int]:
    """Map verified declarations to disjoint MSB-first ports in source order."""
    names: dict[str, tuple[int, ...]] = {}
    values: dict[ValueId, tuple[int, ...]] = {}
    position = 0
    for op in program.body.blocks[0].operations:
        if op.name == "qreg.declare":
            value = op.results[0]
            ports = tuple(range(position, position + register_width(value)))
            names[op.attributes["name"]] = ports
            values[value.id] = ports
            position += len(ports)
    return names, values, position


def verify_register_program(
    program: HybridProgram,
    functions: Sequence[HybridProgram] = (),
    *,
    max_qubits: int = 256,
    max_nodes: int = 4096,
) -> HybridProgram:
    """Verify fresh outputs, initialization, effect order and borrowed arithmetic.

    Every declared register must be returned initialized, once and in declaration
    order. Gates address numeric bit indices (zero is least significant).
    Compute initializes a fresh destination using a verified arithmetic function;
    repeated actual inputs borrow one register and do not copy quantum state.
    """
    _positive_integer(max_qubits, "max_qubits")
    _positive_integer(max_nodes, "max_nodes")
    if max_qubits > 256 or max_nodes > 4096:
        _fail("register.budget", "safety caps are 256 qubits and 4096 nodes")
    if not isinstance(program, HybridProgram) or any(
        not isinstance(f, HybridProgram) for f in functions
    ):
        raise TypeError("use the existing HybridProgram records")
    if len(program.body.blocks) != 1:
        _fail("register.blocks", "register construction requires one ordered block")
    block = program.body.blocks[0]
    if len(block.arguments) != 1 or block.arguments[0].type != QUANTUM_EFFECT:
        _fail("register.entry", "fresh construction requires one quantum effect input")
    if not block.operations or block.operations[-1].name != "qreg.return":
        _fail("register.return", "register construction requires a final qreg.return")
    nodes = sum(
        len(b.arguments) + len(b.operations)
        for f in (program, *functions)
        for b in f.body.blocks
    )
    if nodes > max_nodes:
        _fail("register.budget", "register and expression source exceeds node budget")
    if functions:
        verify_construction(
            functions,
            entry=functions[0].name,
            max_width=max_qubits,
            max_nodes=max_nodes,
        )
    symbols = {f.name: f for f in functions}
    effect = block.arguments[0]
    known: dict[ValueId, Value] = {effect.id: effect}
    registers: list[Value] = []
    initialized: set[ValueId] = set()
    labels: set[str] = set()
    width = 0
    for position, op in enumerate(block.operations):
        attributes = _ATTRIBUTES.get(op.name)
        if attributes is None or set(op.attributes) != attributes or op.regions:
            _fail(
                "register.operation", "unsupported register operation or attributes", op
            )
        if any(known.get(v.id) != v for v in op.operands):
            _fail("register.use", "undefined or mistyped operand", op)
        if op.name == "qreg.declare":
            if op.operands or len(op.results) != 1:
                _fail(
                    "register.declare", "declaration requires one register result", op
                )
            value = op.results[0]
            width += register_width(value, op)
            label = op.attributes["name"]
            if not isinstance(label, str) or not label.strip() or label in labels:
                _fail("register.name", "output names must be nonempty and unique", op)
            if width > max_qubits:
                _fail(
                    "register.budget",
                    "live output widths exceed construction capacity",
                    op,
                )
            registers.append(value)
            labels.add(label)
        else:
            if not op.operands or op.operands[0] != effect:
                _fail("register.effect", "consume the current effect exactly once", op)
            operands = op.operands[1:]
            if any(v not in registers for v in operands):
                _fail(
                    "register.operand",
                    "effects require declared register references",
                    op,
                )
            if op.name == "qreg.return":
                if (
                    position != len(block.operations) - 1
                    or op.results
                    or not registers
                    or operands != tuple(registers)
                    or initialized != {v.id for v in registers}
                ):
                    _fail(
                        "register.return",
                        "return every initialized register once in order",
                        op,
                    )
                continue
            if len(op.results) != 1 or op.results[0].type != QUANTUM_EFFECT:
                _fail(
                    "register.effect", "operation must produce one quantum effect", op
                )
            if not operands:
                _fail(
                    "register.signature",
                    "operation requires a destination register",
                    op,
                )
            destination, *arguments = operands
            if op.name == "qreg.gate":
                index = op.attributes["index"]
                if (
                    arguments
                    or not isinstance(op.attributes["gate"], str)
                    or op.attributes["gate"] not in _GATES
                    or type(index) is not int
                    or not 0 <= index < register_width(destination)
                ):
                    _fail(
                        "register.gate",
                        "use X/H/S/Z and an in-range numeric bit index",
                        op,
                    )
                if destination.id not in initialized:
                    _fail(
                        "register.uninitialized", "gate reads uninitialized storage", op
                    )
            else:
                if destination.id in initialized:
                    _fail(
                        "register.initialized", "destination is already initialized", op
                    )
                if op.name in {"qreg.allocate", "qreg.initialize"}:
                    if arguments:
                        _fail(
                            "register.signature",
                            "initialization has no register inputs",
                            op,
                        )
                    if op.name == "qreg.initialize":
                        constant = op.attributes["value"]
                        if (
                            type(constant) is not int
                            or constant < 0
                            or constant.bit_length() > register_width(destination)
                        ):
                            _fail(
                                "register.constant",
                                "constant must fit its unsigned output",
                                op,
                            )
                else:
                    callee = op.attributes["callee"]
                    if not isinstance(callee, str) or callee not in symbols:
                        _fail(
                            "register.callee",
                            "compute requires a verified expression function",
                            op,
                        )
                    body = symbols[callee].body.blocks[0]
                    if len(arguments) != len(body.arguments) or any(
                        register_width(actual) != _width(formal.type, max_qubits)
                        for actual, formal in zip(
                            arguments, body.arguments, strict=True
                        )
                    ):
                        _fail(
                            "register.signature",
                            "register widths disagree with arithmetic inputs",
                            op,
                        )
                    if destination in arguments:
                        _fail(
                            "register.alias",
                            "fresh destination cannot alias an input",
                            op,
                        )
                    if any(v.id not in initialized for v in arguments):
                        _fail(
                            "register.uninitialized",
                            "arithmetic reads uninitialized storage",
                            op,
                        )
                    returned = body.operations[-1].operands[0]
                    if register_width(destination) < _width(returned.type, max_qubits):
                        _fail(
                            "register.width",
                            "fresh assignment cannot truncate its result",
                            op,
                        )
                initialized.add(destination.id)
            effect = op.results[0]
        for result in op.results:
            if result.id in known:
                _fail("register.ssa", "duplicate register/effect definition", op)
            known[result.id] = result
    return program
