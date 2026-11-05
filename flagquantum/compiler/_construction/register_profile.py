"""Fresh register initialization and ordered effects over existing hybrid records.

Register references denote storage, while quantum.uint values remain immutable
arithmetic expressions. One linear effect orders uses; no external state is
inspected or certified by this structural verifier.
"""

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch

from ...core._composition import _is_finite_unitary
from .._hybrid.model import (
    QUANTUM_EFFECT,
    HybridProgram,
    IRType,
    Operation,
    Value,
    ValueId,
)
from .pauli_rotation import checked_support
from .profile import _fail, _positive_integer, _width, verify_construction
from .register_expansion import expand_register_calls, is_register_function

MAX_PREPARED_AMPLITUDES = 256
MAX_TRANSFORM_WIDTH = 4

_ATTRIBUTES = MappingProxyType(
    {
        "qreg.declare": frozenset({"name"}),
        "qreg.hidden": frozenset({"name"}),
        "qreg.drop": frozenset(),
        "qreg.modular_exp": frozenset({"modulus", "base"}),
        "qreg.control_qft": frozenset(),
        "qreg.local": frozenset({"name"}),
        "qreg.release": frozenset(),
        "qreg.uniform_interval": frozenset({"start", "stop"}),
        "qreg.allocate": frozenset(),
        "qreg.initialize": frozenset({"value"}),
        "qreg.prepare_ghz": frozenset(),
        "qreg.prepare_amplitudes": frozenset({"amplitudes", "bound"}),
        "qreg.inplace_prepare_amplitudes": frozenset({"amplitudes", "bound"}),
        "qreg.unitary": frozenset({"matrix"}),
        "qreg.qft": frozenset(),
        "qreg.inverse_qft": frozenset(),
        "qreg.type1_transform": frozenset(),
        "qreg.type2_transform": frozenset({"kind"}),
        "qreg.rzz": frozenset({"angle"}),
        "qreg.pauli_rotation": frozenset({"support", "angle"}),
        "qreg.pair_gate": frozenset({"gate"}),
        "qreg.control_h": frozenset(),
        "qreg.control_x": frozenset(),
        "qreg.array_phase_flip": frozenset(),
        "qreg.control_phase": frozenset({"angle"}),
        "qreg.gate": frozenset({"gate", "index"}),
        "qreg.compute": frozenset({"callee"}),
        "qreg.uncompute": frozenset(),
        "qreg.phase": frozenset({"callee"}),
        "qreg.control_call": frozenset({"callee"}),
        "qreg.call": frozenset({"callee"}),
        "qreg.return": frozenset(),
    }
)
_ANGLED_PAIR_GATES = frozenset({"crx", "cry", "crz", "rzz"})
_PAIR_GATES = frozenset({"cx", "cz", "swap"}) | _ANGLED_PAIR_GATES
_ANGLE_GATES = frozenset({"phase", "rx", "ry", "rz"})
_GATES = frozenset({"x", "h", "s", "z", "i", "sx"}) | _ANGLE_GATES
MAX_MATRIX_QUBITS = 4


def unitary_matrix(entries: object, width: int, op: Operation) -> torch.Tensor:
    """Decode a bounded dense matrix from immutable real/imaginary pairs."""
    if not 1 <= width <= MAX_MATRIX_QUBITS:
        _fail("register.unitary", "matrix targets must have one to four qubits", op)
    size = 1 << width
    if (
        not isinstance(entries, tuple)
        or len(entries) != size
        or any(not isinstance(row, tuple) or len(row) != size for row in entries)
        or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or any(type(part) is not float or not math.isfinite(part) for part in entry)
            for row in entries
            for entry in row
        )
    ):
        _fail(
            "register.unitary",
            "expected a finite square matrix matching the target width",
            op,
        )
    return torch.view_as_complex(torch.tensor(entries, dtype=torch.float64))


def _check_real_preparation(op: Operation, destination: Value) -> None:
    amplitudes = op.attributes["amplitudes"]
    bound = op.attributes["bound"]
    if (
        not isinstance(amplitudes, tuple)
        or not 2 <= len(amplitudes) <= MAX_PREPARED_AMPLITUDES
        or len(amplitudes) & (len(amplitudes) - 1)
        or len(amplitudes).bit_length() - 1 != register_width(destination)
        or any(
            type(value) is not float or not math.isfinite(value) for value in amplitudes
        )
        or not math.isclose(math.hypot(*amplitudes), 1.0, rel_tol=0, abs_tol=1e-12)
        or type(bound) is not float
        or not math.isfinite(bound)
        or bound < 0
    ):
        _fail(
            "register.amplitudes",
            "preparation requires a normalized finite real vector matching the target and a nonnegative finite bound",
            op,
        )


def equatorial_rotation(theta: float, phi: float) -> tuple[tuple[str, float], ...]:
    """Return the exact single-qubit rotation in execution order."""
    return (("rz", -phi), ("rx", theta), ("rz", phi))


def euler_rotation(
    theta: float, phi: float, lam: float
) -> tuple[tuple[str, float], ...]:
    """Return the non-global part of the U gate in execution order."""
    return (("phase", lam), *equatorial_rotation(theta, math.pi / 2), ("phase", phi))


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


RecordField = tuple[str, str, int] | tuple[str, str, int, int]


def record_fields(op: Operation) -> tuple[RecordField, ...]:
    """Validate a flat record's owned output layout before projecting readout names."""
    if "fields" not in op.attributes and "record_type" not in op.attributes:
        return ()
    fields = op.attributes.get("fields")
    record_type = op.attributes.get("record_type")
    if (
        not isinstance(record_type, str)
        or not record_type.isidentifier()
        or not isinstance(fields, tuple)
        or not fields
    ):
        _fail("register.record", "record requires a type and nonempty field layout", op)
    names: set[str] = set()
    width = 0
    for field in fields:
        if not isinstance(field, tuple) or len(field) not in {3, 4}:
            _fail("register.record", "record fields require name, kind and width", op)
        name, kind, size = field[:3]
        length = field[3] if len(field) == 4 else 1
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in names
            or name in {"size", "len"}
            or kind not in ("qbit", "qbit[]", "qnum", "qnum[]", "qbit[][]")
            or type(size) is not int
            or not 1 <= size <= 256
            or (kind == "qbit" and size != 1)
            or type(length) is not int
            or not 1 <= length <= 256
            or (kind in {"qnum[]", "qbit[][]"}) != (len(field) == 4)
            or size * length > 256
        ):
            _fail(
                "register.record",
                "record fields must be unique bounded unsigned fields",
                op,
            )
        names.add(name)
        width += size * length
    if (
        width > 256
        or len(op.results) != 1
        or width != register_width(op.results[0], op)
    ):
        _fail("register.record", "record fields must cover the register exactly", op)
    return fields


def record_readouts(fields: tuple[RecordField, ...]) -> tuple[tuple[str, int], ...]:
    """Project disjoint fields, numeric elements and bit-array rows for readout."""
    return tuple(
        (f"{field[0]}[{index}]" if len(field) == 4 else field[0], field[2])
        for field in fields
        for index in range(field[3] if len(field) == 4 else 1)
    )


def register_ports(
    program: HybridProgram,
) -> tuple[dict[str, tuple[int, ...]], dict[ValueId, tuple[int, ...]], int]:
    """Map verified outputs first, then reuse zero local slots by lifetime."""
    names: dict[str, tuple[int, ...]] = {}
    values: dict[ValueId, tuple[int, ...]] = {}
    position = 0
    for op in program.body.blocks[0].operations:
        if op.name == "qreg.declare":
            value = op.results[0]
            ports = tuple(range(position, position + register_width(value)))
            fields = record_fields(op)
            if fields:
                offset = 0
                for name, size in record_readouts(fields):
                    names[f"{op.attributes['name']}.{name}"] = ports[
                        len(ports) - offset - size : len(ports) - offset
                    ]
                    offset += size
            else:
                names[op.attributes["name"]] = ports
            values[value.id] = ports
            position += len(ports)
    following = position
    for op in program.body.blocks[0].operations:
        if op.name == "qreg.hidden":
            value = op.results[0]
            end = following + register_width(value)
            values[value.id] = tuple(range(following, end))
            following = end
    free: set[int] = set()
    for op in program.body.blocks[0].operations:
        if op.name == "qreg.local":
            value = op.results[0]
            width = register_width(value)
            reused = tuple(sorted(free)[:width])
            free.difference_update(reused)
            count = width - len(reused)
            values[value.id] = (*reused, *range(following, following + count))
            following += count
        elif op.name == "qreg.release":
            free.update(values[op.operands[1].id])
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
    Phase calls borrow initialized values without changing basis values or roles.
    Uncompute makes unchanged directly computed storage fresh, retaining ownership.
    Local storage must be released fresh before return and cannot escape as output.
    """
    _positive_integer(max_qubits, "max_qubits")
    _positive_integer(max_nodes, "max_nodes")
    if max_qubits > 256 or max_nodes > 4096:
        _fail("register.budget", "safety caps are 256 qubits and 4096 nodes")
    if not isinstance(program, HybridProgram) or any(
        not isinstance(f, HybridProgram) for f in functions
    ):
        raise TypeError("use the existing HybridProgram records")
    nodes = sum(
        len(b.arguments) + len(b.operations)
        for f in (program, *functions)
        for b in f.body.blocks
    )
    if nodes > max_nodes:
        _fail("register.budget", "register and expression source exceeds node budget")
    if len({f.name for f in functions}) != len(functions):
        _fail("register.callee", "function definitions require unique names")
    arithmetic = tuple(f for f in functions if not is_register_function(f))
    helpers = {f.name: f for f in functions if is_register_function(f)}
    if arithmetic:
        verify_construction(
            arithmetic,
            entry=arithmetic[0].name,
            max_width=max_qubits,
            max_nodes=max_nodes,
        )
    active: set[str] = set()
    depths: dict[str, int] = {}

    def visit(name: str) -> int:
        if name in active:
            _fail("register.recursion", f"register call cycle at {name!r}")
        if name not in depths:
            if len(active) >= 32:
                _fail("register.budget", "register call nesting exceeds 32")
            active.add(name)
            depth = 1
            for op in helpers[name].body.blocks[0].operations:
                if op.name in {"qreg.call", "qreg.control_call"}:
                    callee = op.attributes.get("callee")
                    if not isinstance(callee, str) or callee not in helpers:
                        _fail(
                            "register.callee", "call requires a register function", op
                        )
                    depth = max(depth, 1 + visit(callee))
            if depth > 32:
                _fail("register.budget", "register call nesting exceeds 32")
            _verify_body(
                helpers[name], arithmetic, helpers, max_qubits, const_entry=True
            )
            active.remove(name)
            depths[name] = depth
        return depths[name]

    for name in helpers:
        visit(name)
    _verify_body(program, arithmetic, helpers, max_qubits)
    expanded = expand_register_calls(program, functions, max_nodes=max_nodes)
    if expanded is not program:
        _verify_body(expanded, arithmetic, {}, max_qubits)
    return program


def _verify_body(
    program: HybridProgram,
    arithmetic: Sequence[HybridProgram],
    helpers: Mapping[str, HybridProgram],
    max_qubits: int,
    *,
    const_entry: bool = False,
) -> None:
    if len(program.body.blocks) != 1:
        _fail("register.blocks", "register construction requires one ordered block")
    block = program.body.blocks[0]
    if (
        not block.arguments
        or block.arguments[0].type != QUANTUM_EFFECT
        or (not const_entry and len(block.arguments) != 1)
    ):
        _fail(
            "register.entry",
            "fresh roots require one effect; callees may add Const references",
        )
    if not block.operations or block.operations[-1].name != "qreg.return":
        _fail("register.return", "register construction requires a final qreg.return")
    symbols = {f.name: f for f in arithmetic}
    effect = block.arguments[0]
    known: dict[ValueId, Value] = {effect.id: effect}
    borrowed = block.arguments[1:]
    for value in borrowed:
        register_width(value)
        if value.id in known:
            _fail("register.ssa", "duplicate Const entry reference")
        known[value.id] = value
    registers: list[Value] = list(borrowed)
    returned: list[Value] = list(borrowed)
    local_storage: set[Value] = set()
    initialized = {v.id for v in borrowed}
    reversible: dict[ValueId, frozenset[ValueId]] = {}
    labels: set[str] = set()
    readout_labels: set[str] = set()
    width = sum(register_width(v) for v in borrowed)
    if width > max_qubits:
        _fail("register.budget", "Const entry widths exceed construction capacity")

    def check_call(
        op: Operation, arguments: Sequence[Value], destination: Value | None = None
    ) -> None:
        callee = op.attributes["callee"]
        if not isinstance(callee, str) or callee not in symbols:
            _fail("register.callee", "call requires a verified expression function", op)
        body = symbols[callee].body.blocks[0]
        if len(arguments) != len(body.arguments) or any(
            register_width(actual) != _width(formal.type, max_qubits)
            for actual, formal in zip(arguments, body.arguments, strict=True)
        ):
            _fail(
                "register.signature",
                "register widths disagree with arithmetic inputs",
                op,
            )
        if destination is not None and destination in arguments:
            _fail("register.alias", "fresh destination cannot alias an input", op)
        if any(v.id not in initialized for v in arguments):
            _fail("register.uninitialized", "call reads uninitialized storage", op)
        returned = body.operations[-1].operands[0]
        result_width = _width(returned.type, max_qubits)
        if destination is None:
            if "angle" not in op.attributes and result_width != 1:
                _fail(
                    "register.phase",
                    "phase call requires an exactly one-bit return",
                    op,
                )
        elif register_width(destination) < result_width:
            _fail("register.width", "fresh assignment cannot truncate its result", op)

    hidden: set[Value] = set()
    for position, op in enumerate(block.operations):
        attributes = _ATTRIBUTES.get(op.name)
        if (
            op.name == "qreg.gate"
            and isinstance(op.attributes.get("gate"), str)
            and op.attributes["gate"] in _ANGLE_GATES
        ):
            assert attributes is not None
            attributes = attributes | {"angle"}
        if (
            op.name
            in {"qreg.gate", "qreg.control_qft", "qreg.inplace_prepare_amplitudes"}
            and "inverse" in op.attributes
        ):
            if op.attributes["inverse"] is not True:
                _fail("register.adjoint", "inverse marker must be true", op)
            assert attributes is not None
            attributes = attributes | {"inverse"}
        if op.name == "qreg.control_qft" and "control_value" in op.attributes:
            assert attributes is not None
            attributes = attributes | {"control_value"}
        if op.name == "qreg.pauli_rotation" and "control_index" in op.attributes:
            assert attributes is not None
            attributes = attributes | {"control_index"}
        if op.name == "qreg.pair_gate":
            assert attributes is not None
            if (
                isinstance(gate := op.attributes.get("gate"), str)
                and gate in _ANGLED_PAIR_GATES
            ):
                attributes = attributes | {"angle"}
            if "indices" in op.attributes:
                attributes = attributes | {"indices"}
        if (
            op.name in {"qreg.control_x", "qreg.control_phase"}
            and "indices" in op.attributes
        ):
            assert attributes is not None
            attributes = attributes | {"indices"}
        if op.name == "qreg.declare" and (
            "fields" in op.attributes or "record_type" in op.attributes
        ):
            assert attributes is not None
            attributes = attributes | {"fields", "record_type"}
        if op.name == "qreg.hidden" and "workspace" in op.attributes:
            if op.attributes["workspace"] is not True:
                _fail(
                    "register.workspace",
                    "hidden workspace marker must be exactly True",
                    op,
                )
            assert attributes is not None
            attributes = attributes | {"workspace"}
        if op.name == "qreg.unitary":
            assert attributes is not None
            attributes = attributes | (
                op.attributes.keys() & {"indices", "control_index", "power"}
            )
        if op.name == "qreg.phase" and "angle" in op.attributes:
            angle = op.attributes["angle"]
            if type(angle) is not float or not math.isfinite(angle):
                _fail("register.phase", "phase angle must be a finite float", op)
            attributes = (
                frozenset({"angle", "callee"})
                if len(op.operands) > 1
                else frozenset({"angle"})
            )
        if op.name == "qreg.call" and "inverse" in op.attributes:
            if op.attributes["inverse"] is not True:
                _fail("register.adjoint", "inverse call marker must be true", op)
            attributes = frozenset({"callee", "inverse"})
        if attributes is None or set(op.attributes) != attributes or op.regions:
            _fail(
                "register.operation", "unsupported register operation or attributes", op
            )
        if any(known.get(v.id) != v for v in op.operands):
            _fail("register.use", "undefined or mistyped operand", op)
        if op.name in {"qreg.declare", "qreg.local", "qreg.hidden"}:
            if op.operands or len(op.results) != 1:
                _fail(
                    "register.declare", "declaration requires one register result", op
                )
            value = op.results[0]
            width += register_width(value, op)
            label = op.attributes["name"]
            if not isinstance(label, str) or not label.strip() or label in labels:
                _fail("register.name", "register names must be nonempty and unique", op)
            if width > max_qubits:
                _fail(
                    "register.budget",
                    "live register widths exceed construction capacity",
                    op,
                )
            registers.append(value)
            if op.name == "qreg.declare":
                fields = record_fields(op)
                projected = (
                    tuple(f"{label}.{name}" for name, _ in record_readouts(fields))
                    if fields
                    else (label,)
                )
                if any(name in readout_labels for name in projected):
                    _fail("register.name", "projected output names must be unique", op)
                readout_labels.update(projected)
                returned.append(value)
            elif op.name == "qreg.local":
                local_storage.add(value)
            else:
                hidden.add(value)
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
                if local_storage:
                    _fail("register.local", "release every local before return", op)
                if hidden:
                    _fail(
                        "register.drop", "drop every hidden register before return", op
                    )
                if (
                    position != len(block.operations) - 1
                    or op.results
                    or not registers
                    or operands != tuple(returned)
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
            if not operands and op.name == "qreg.phase" and "angle" in op.attributes:
                effect = op.results[0]
                if effect.id in known:
                    _fail("register.ssa", "duplicate register/effect definition", op)
                known[effect.id] = effect
                continue
            if not operands:
                _fail(
                    "register.signature",
                    "operation requires at least one register operand",
                    op,
                )
            destination, *arguments = operands
            changed: tuple[Value, ...] = ()
            if op.name in {"qreg.call", "qreg.control_call"}:
                controlled = op.name == "qreg.control_call"
                if controlled:
                    if (
                        register_width(destination) != 1
                        or destination.id not in initialized
                    ):
                        _fail(
                            "register.control",
                            "controlled call requires an initialized bit",
                            op,
                        )
                    if destination in arguments or len(set(arguments)) != len(
                        arguments
                    ):
                        _fail(
                            "register.alias",
                            "controlled call arguments must be disjoint",
                            op,
                        )
                    operands = tuple(arguments)
                callee = op.attributes["callee"]
                if not isinstance(callee, str) or callee not in helpers:
                    _fail("register.callee", "call requires a register function", op)
                body = helpers[callee].body.blocks[0]
                const = body.arguments[1:]
                outputs = tuple(
                    q.results[0] for q in body.operations if q.name == "qreg.declare"
                )
                if controlled and outputs:
                    _fail(
                        "register.control",
                        "controlled calls require a Const-only callee",
                        op,
                    )
                if op.attributes.get("inverse") and outputs:
                    _fail(
                        "register.adjoint",
                        "inverse call requires a Const-only callee",
                        op,
                    )
                formal = (*const, *outputs)
                if len(operands) != len(formal) or any(
                    actual.type != expected.type
                    for actual, expected in zip(operands, formal, strict=True)
                ):
                    _fail("register.signature", "call roles or widths disagree", op)
                reads, writes = operands[: len(const)], operands[len(const) :]
                if any(v.id not in initialized for v in reads):
                    _fail(
                        "register.uninitialized", "Const actual is not initialized", op
                    )
                if len(set(writes)) != len(writes) or set(writes).intersection(reads):
                    _fail(
                        "register.alias",
                        "Output actuals must be disjoint from all arguments",
                        op,
                    )
                if any(v in borrowed for v in writes):
                    _fail("register.const", "Const cannot be passed as Output", op)
                if any(v.id in initialized for v in writes):
                    _fail(
                        "register.initialized", "call requires fresh Output actuals", op
                    )
                initialized.update(v.id for v in writes)
                changed = tuple(writes)
            elif op.name == "qreg.unitary":
                data = operands
                if "control_index" in op.attributes:
                    index = op.attributes["control_index"]
                    if (
                        len(operands) < 2
                        or type(index) is not int
                        or not 0 <= index < register_width(operands[-1])
                    ):
                        _fail(
                            "register.unitary",
                            "unitary control index must be in bounds",
                            op,
                        )
                    data = operands[:-1]
                    if operands[-1] in data:
                        _fail(
                            "register.unitary",
                            "unitary control and data storage must be disjoint",
                            op,
                        )
                if "power" in op.attributes:
                    power = op.attributes["power"]
                    if type(power) is not int or not 1 <= power <= 128:
                        _fail(
                            "register.unitary",
                            "unitary power must be an integer in 1..128",
                            op,
                        )
                if "indices" in op.attributes:
                    indices = op.attributes["indices"]
                    if (
                        not isinstance(indices, tuple)
                        or len(indices) != len(data)
                        or not 1 <= len(indices) <= MAX_MATRIX_QUBITS
                        or any(
                            type(index) is not int
                            or not 0 <= index < register_width(value)
                            for value, index in zip(data, indices, strict=True)
                        )
                        or len(set(zip(data, indices, strict=True))) != len(indices)
                    ):
                        _fail(
                            "register.unitary",
                            "unitary indices must be in bounds and distinct",
                            op,
                        )
                    width = len(indices)
                else:
                    if len(data) != 1:
                        _fail(
                            "register.unitary",
                            "unitary requires one writable register",
                            op,
                        )
                    width = register_width(destination)
                if any(value in borrowed for value in data):
                    _fail("register.unitary", "unitary requires writable qubits", op)
                if any(value.id not in initialized for value in operands):
                    _fail(
                        "register.uninitialized",
                        "unitary requires initialized qubits",
                        op,
                    )
                matrix = unitary_matrix(op.attributes["matrix"], width, op)
                if not _is_finite_unitary(matrix):
                    _fail("register.unitary", "matrix must be unitary", op)
                changed = tuple(data)
            elif op.name in {"qreg.qft", "qreg.inverse_qft"}:
                if arguments:
                    _fail("register.transform", "QFT requires one register", op)
                if destination in borrowed:
                    _fail("register.const", "QFT requires a writable register", op)
                if destination.id not in initialized:
                    _fail(
                        "register.uninitialized",
                        "QFT requires an initialized register",
                        op,
                    )
                changed = (destination,)
            elif op.name in {"qreg.type1_transform", "qreg.type2_transform"}:
                type1 = op.name == "qreg.type1_transform"
                kind = "type1" if type1 else op.attributes["kind"]
                if (
                    not isinstance(kind, str)
                    or (not type1 and kind not in {"cosine", "sine", "combined"})
                    or len(arguments) != int(kind == "combined")
                    or (arguments and register_width(arguments[0]) != 1)
                    or not 1 <= register_width(destination) <= MAX_TRANSFORM_WIDTH
                ):
                    _fail(
                        "register.transform",
                        (
                            "invalid bounded Type I transform operands"
                            if type1
                            else "invalid bounded Type II transform operands"
                        ),
                        op,
                    )
                if destination in borrowed or destination in arguments:
                    _fail(
                        "register.transform",
                        "transform requires a writable target disjoint from its selector",
                        op,
                    )
                if any(value.id not in initialized for value in operands):
                    _fail(
                        "register.uninitialized",
                        "transform requires initialized registers",
                        op,
                    )
                changed = (destination,)
            elif op.name == "qreg.array_phase_flip":
                if arguments or destination.id not in initialized:
                    _fail(
                        "register.control",
                        "phase flip requires one initialized register",
                        op,
                    )
            elif op.name == "qreg.control_phase" and "indices" in op.attributes:
                indices = op.attributes["indices"]
                angle = op.attributes["angle"]
                if (
                    not isinstance(indices, tuple)
                    or len(indices) != len(operands)
                    or not 1 <= len(indices) <= 8
                    or any(
                        type(index) is not int or not 0 <= index < register_width(value)
                        for value, index in zip(operands, indices, strict=True)
                    )
                    or len(set(zip(operands, indices, strict=True))) != len(indices)
                ):
                    _fail(
                        "register.control",
                        "phase indices must be bounded and distinct",
                        op,
                    )
                if type(angle) is not float or not math.isfinite(angle):
                    _fail("register.control", "phase angle must be finite", op)
                if any(value.id not in initialized for value in operands):
                    _fail(
                        "register.uninitialized",
                        "phase requires initialized qubits",
                        op,
                    )
            elif op.name == "qreg.pauli_rotation":
                checked_support(
                    op.attributes["support"], register_width(destination), op
                )
                angle = op.attributes["angle"]
                controlled = "control_index" in op.attributes
                if (
                    type(angle) is not float
                    or not math.isfinite(angle)
                    or not math.isfinite(2 * angle)
                ):
                    _fail(
                        "register.pauli",
                        "Pauli angle and doubled angle must be finite floats",
                        op,
                    )
                if len(arguments) != int(controlled):
                    _fail(
                        "register.pauli",
                        "Pauli rotation needs a target and optional control",
                        op,
                    )
                if any(value.id not in initialized for value in operands):
                    _fail(
                        "register.uninitialized",
                        "Pauli rotation requires initialized storage",
                        op,
                    )
                if destination in borrowed:
                    _fail("register.const", "Pauli target must be writable", op)
                if controlled:
                    index = op.attributes["control_index"]
                    if (
                        arguments[0] == destination
                        or type(index) is not int
                        or not 0 <= index < register_width(arguments[0])
                    ):
                        _fail(
                            "register.pauli",
                            "Pauli control requires a disjoint register and bounded index",
                            op,
                        )
                changed = (destination,)
            elif op.name == "qreg.control_x" and "indices" in op.attributes:
                indices = op.attributes["indices"]
                if (
                    not isinstance(indices, tuple)
                    or len(indices) != len(operands)
                    or len(indices) < 2
                    or any(
                        type(index) is not int or not 0 <= index < register_width(value)
                        for value, index in zip(operands, indices, strict=True)
                    )
                    or len(set(zip(operands, indices, strict=True))) != len(indices)
                ):
                    _fail(
                        "register.control",
                        "control indices must be in bounds and distinct",
                        op,
                    )
                if any(value.id not in initialized for value in operands):
                    _fail(
                        "register.uninitialized",
                        "control requires initialized qubits",
                        op,
                    )
                if operands[-1] in borrowed:
                    _fail("register.const", "controlled target must be writable", op)
                changed = (operands[-1],)
            elif op.name in {"qreg.control_h", "qreg.control_x", "qreg.control_phase"}:
                if (
                    len(arguments) > 1
                    or (
                        op.name != "qreg.control_x" and register_width(destination) != 1
                    )
                    or any(value.id not in initialized for value in operands)
                ):
                    _fail(
                        "register.control",
                        "control requires an initialized condition bit and an initialized target",
                        op,
                    )
                if destination in arguments:
                    _fail(
                        "register.control", "condition and target must be disjoint", op
                    )
                if op.name in {"qreg.control_h", "qreg.control_x"}:
                    if (
                        len(arguments) != 1
                        or arguments[0] in borrowed
                        or register_width(arguments[0]) != 1
                    ):
                        _fail(
                            "register.control",
                            "controlled gate requires a writable scalar target",
                            op,
                        )
                    changed = tuple(arguments)
                else:
                    angle = op.attributes["angle"]
                    if type(angle) is not float or not math.isfinite(angle):
                        _fail("register.control", "phase requires a finite angle", op)
                    if arguments:
                        width = register_width(arguments[0])
                        if width > 8:
                            _fail(
                                "register.control",
                                "phase target exceeds eight bits",
                                op,
                            )
                        try:
                            math.ldexp(angle, width - 1)
                        except OverflowError:
                            _fail(
                                "register.control",
                                "weighted phase angle exceeds finite range",
                                op,
                            )
            elif op.name == "qreg.phase":
                check_call(op, operands)
            elif op.name == "qreg.modular_exp":
                modulus, base = op.attributes["modulus"], op.attributes["base"]
                if (
                    len(arguments) != 1
                    or destination in borrowed
                    or destination in arguments
                    or any(v.id not in initialized for v in operands)
                    or not 1 <= register_width(arguments[0]) <= 8
                    or not 1 <= register_width(destination) <= 4
                ):
                    _fail(
                        "register.modular_exp",
                        "modular exponentiation requires distinct initialized registers within width bounds",
                        op,
                    )
                if (
                    type(modulus) is not int
                    or type(base) is not int
                    or not 2 <= modulus <= 1 << register_width(destination)
                    or base < 1
                    or math.gcd(base, modulus) != 1
                ):
                    _fail(
                        "register.modular_exp",
                        "base must be coprime to a modulus fitting the target",
                        op,
                    )
                changed = (destination,)
            elif op.name == "qreg.control_qft":
                control_width = register_width(destination)
                if (
                    len(arguments) != 1
                    or destination in arguments
                    or arguments[0] in borrowed
                    or any(v.id not in initialized for v in operands)
                ):
                    _fail(
                        "register.control",
                        "controlled QFT needs an initialized selector and disjoint writable target",
                        op,
                    )
                if "control_value" in op.attributes:
                    control_value = op.attributes["control_value"]
                    if (
                        not 1 <= control_width <= 8
                        or not 1 <= register_width(arguments[0]) <= 8
                        or type(control_value) is not int
                        or not 0 <= control_value < (1 << control_width)
                    ):
                        _fail(
                            "register.control",
                            "controlled QFT basis word or register width is out of range",
                            op,
                        )
                elif control_width != 1:
                    _fail(
                        "register.control",
                        "controlled QFT without a basis word needs a control bit",
                        op,
                    )
                changed = tuple(arguments)
            elif op.name == "qreg.inplace_prepare_amplitudes":
                if (
                    arguments
                    or destination in borrowed
                    or destination.id not in initialized
                ):
                    _fail(
                        "register.amplitudes",
                        "inplace preparation requires one initialized writable register",
                        op,
                    )
                _check_real_preparation(op, destination)
                changed = (destination,)
            elif op.name == "qreg.drop":
                if (
                    arguments
                    or destination not in hidden
                    or destination.id not in initialized
                ):
                    _fail(
                        "register.drop", "drop requires initialized hidden storage", op
                    )
                hidden.remove(destination)
                registers.remove(destination)
                initialized.remove(destination.id)
                changed = (destination,)
            elif op.name == "qreg.release":
                if destination in borrowed:
                    _fail("register.const", "Const cannot be released", op)
                if arguments or destination not in local_storage:
                    _fail("register.release", "release requires one owned local", op)
                if destination.id in initialized:
                    _fail(
                        "register.initialized",
                        "release requires fresh zero storage",
                        op,
                    )
                registers.remove(destination)
                local_storage.remove(destination)
                width -= register_width(destination)
                changed = (destination,)
            elif op.name == "qreg.uncompute":
                if destination in borrowed:
                    _fail("register.const", "Const cannot be uncomputed", op)
                if arguments or destination.id not in reversible:
                    _fail(
                        "register.uncompute",
                        "cleanup requires an unchanged direct compute and its inputs",
                        op,
                    )
                initialized.remove(destination.id)
                changed = (destination,)
            elif op.name == "qreg.rzz":
                angle = op.attributes["angle"]
                if (
                    arguments
                    or register_width(destination) != 2
                    or type(angle) is not float
                    or not math.isfinite(angle)
                ):
                    _fail(
                        "register.gate",
                        "RZZ requires a two-bit register and finite angle",
                        op,
                    )
                if destination in borrowed:
                    _fail("register.const", "Const cannot be a gate target", op)
                if destination.id not in initialized:
                    _fail(
                        "register.uninitialized", "gate reads uninitialized storage", op
                    )
                changed = (destination,)
            elif op.name == "qreg.pair_gate":
                gate = op.attributes["gate"]
                if (
                    not isinstance(gate, str)
                    or gate not in _PAIR_GATES
                    or len(operands) != 2
                ):
                    _fail(
                        "register.gate",
                        "pair gates require two distinct scalar registers",
                        op,
                    )
                indices = op.attributes.get("indices")
                if "indices" not in op.attributes:
                    if len(set(operands)) != 2 or any(
                        register_width(value) != 1 for value in operands
                    ):
                        _fail(
                            "register.gate",
                            "pair gates require distinct scalar registers",
                            op,
                        )
                elif (
                    not isinstance(indices, tuple)
                    or len(indices) != 2
                    or any(
                        type(index) is not int or not 0 <= index < register_width(value)
                        for value, index in zip(operands, indices, strict=True)
                    )
                    or len(set(zip(operands, indices, strict=True))) != 2
                ):
                    _fail(
                        "register.gate",
                        "pair gate indices must be in bounds and distinct",
                        op,
                    )
                if gate in _ANGLED_PAIR_GATES:
                    angle = op.attributes["angle"]
                    if type(angle) is not float or not math.isfinite(angle):
                        _fail(
                            "register.gate",
                            "gate angle must be a finite float in radians",
                            op,
                        )
                changed = tuple(operands if gate in {"swap", "rzz"} else arguments)
                if any(value in borrowed for value in changed):
                    _fail("register.const", "Const cannot be a gate target", op)
                if any(value.id not in initialized for value in operands):
                    _fail(
                        "register.uninitialized", "gate reads uninitialized storage", op
                    )
            elif op.name == "qreg.uniform_interval":
                start, stop = op.attributes["start"], op.attributes["stop"]
                size = register_width(destination)
                previous = block.operations[position - 1]
                if (
                    arguments
                    or destination in borrowed
                    or previous.name != "qreg.allocate"
                    or previous.operands[1:] != (destination,)
                ):
                    _fail(
                        "register.prepare",
                        "uniform preparation requires a fresh allocation",
                        op,
                    )
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or size > 8
                    or not 0 <= start < stop <= 1 << size
                ):
                    _fail(
                        "register.prepare",
                        "invalid uniform interval or width above eight",
                        op,
                    )
                changed = (destination,)
            elif op.name == "qreg.gate":
                if destination in borrowed:
                    _fail("register.const", "Const cannot be a gate target", op)
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
                        "use X/H/S/Z/phase/rx/rz and an in-range numeric bit index",
                        op,
                    )
                if op.attributes["gate"] in _ANGLE_GATES:
                    angle = op.attributes["angle"]
                    if type(angle) is not float or not math.isfinite(angle):
                        _fail(
                            "register.gate",
                            "gate angle must be a finite float in radians",
                            op,
                        )
                if destination.id not in initialized:
                    _fail(
                        "register.uninitialized", "gate reads uninitialized storage", op
                    )
                changed = (destination,)
            else:
                if destination in borrowed:
                    _fail(
                        "register.const",
                        "Const cannot be initialized or computed into",
                        op,
                    )
                if destination.id in initialized:
                    _fail(
                        "register.initialized", "destination is already initialized", op
                    )
                if op.name in {
                    "qreg.allocate",
                    "qreg.initialize",
                    "qreg.prepare_ghz",
                    "qreg.prepare_amplitudes",
                }:
                    if arguments:
                        _fail(
                            "register.signature",
                            "initialization has no register inputs",
                            op,
                        )
                    if op.name == "qreg.prepare_amplitudes":
                        _check_real_preparation(op, destination)
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
                    check_call(op, arguments, destination)
                initialized.add(destination.id)
                changed = (destination,)
            for key, dependencies in tuple(reversible.items()):
                if any(value.id in dependencies for value in changed):
                    del reversible[key]
            if op.name == "qreg.compute":
                reversible[destination.id] = frozenset(v.id for v in operands)
            effect = op.results[0]
        for result in op.results:
            if result.id in known:
                _fail("register.ssa", "duplicate register/effect definition", op)
            known[result.id] = result
    _, bindings, output_width = register_ports(program)
    reserved = sum(register_width(v) for v in borrowed) + max(
        (max(ports) + 1 for ports in bindings.values()), default=output_width
    )
    if reserved > max_qubits:
        _fail("register.budget", "reserved outputs and local pool exceed capacity")
