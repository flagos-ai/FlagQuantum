"""Fresh register initialization and ordered effects over existing hybrid records.

Register references denote storage, while quantum.uint values remain immutable
arithmetic expressions. One linear effect orders uses; no external state is
inspected or certified by this structural verifier.
"""

import math
from collections.abc import Sequence
from dataclasses import replace
from types import MappingProxyType

import torch

from .._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    Value,
    ValueId,
)
from .profile import _fail, _positive_integer, verify_construction
from .register_expansion import expand_register_calls, is_register_function

MAX_PREPARED_AMPLITUDES = 256
_MAX_INPLACE_AMPLITUDES = 4096
MAX_TRANSFORM_WIDTH = 4

_ATTRIBUTES = MappingProxyType(
    {
        "qreg.declare": frozenset({"name"}),
        "qreg.hidden": frozenset({"name"}),
        "qreg.drop": frozenset(),
        "qreg.publish": frozenset({"destinations", "locations"}),
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
        "qreg.replay": frozenset({"callee", "count"}),
        "qreg.unitary_call": frozenset({"callee"}),
        "qreg.gate_tape": frozenset({"workspace", "templates", "schedule"}),
        "qreg.reset": frozenset({"index"}),
        "qreg.gate": frozenset({"gate", "index"}),
        "qreg.compute": frozenset({"callee"}),
        "qreg.xor": frozenset({"callee"}),
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
MAX_MATRIX_QUBITS = 5


def unitary_matrix(entries: object, width: int, op: Operation) -> torch.Tensor:
    """Decode a bounded dense matrix from immutable real/imaginary pairs."""
    if not 1 <= width <= MAX_MATRIX_QUBITS:
        _fail("register.unitary", "matrix targets must have one to five qubits", op)
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


def _check_real_preparation(op: Operation, width: int) -> None:
    amplitudes = op.attributes["amplitudes"]
    bound = op.attributes["bound"]
    limit = (
        _MAX_INPLACE_AMPLITUDES
        if op.name == "qreg.inplace_prepare_amplitudes"
        and "indices" in op.attributes
        and not op.attributes.get("control_count")
        else MAX_PREPARED_AMPLITUDES
    )
    if (
        not isinstance(amplitudes, tuple)
        or not 2 <= len(amplitudes) <= limit
        or len(amplitudes) & (len(amplitudes) - 1)
        or len(amplitudes).bit_length() - 1 != width
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
    """Validate disjoint record leaves before projecting their readout names."""
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
            or len(name.split(".")) > 32
            or any(
                not part.isidentifier() or part in {"size", "len"}
                for part in name.split(".")
            )
            or any(
                name == previous
                or name.startswith(previous + ".")
                or previous.startswith(name + ".")
                for previous in names
            )
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


def _publication(program: HybridProgram) -> Operation | None:
    operations = program.body.blocks[0].operations
    if len(operations) >= 2 and operations[-2].name == "qreg.publish":
        return operations[-2]
    return None


def _publication_destinations(op: Operation) -> tuple[Value, ...]:
    count = op.attributes.get("destinations")
    if type(count) is not int or not 1 <= count < len(op.operands) - 1:
        _fail("register.publish", "publication requires outputs and hidden storage", op)
    return op.operands[1 : 1 + count]


def _publication_sources(op: Operation) -> tuple[Value, ...]:
    return op.operands[1 + len(_publication_destinations(op)) :]


def _publication_layout(op: Operation) -> tuple[tuple[Value, int], ...]:
    sources = _publication_sources(op)
    locations = op.attributes.get("locations")
    if (
        not isinstance(locations, tuple)
        or len(locations) != sum(register_width(source) for source in sources)
        or len(locations)
        != sum(register_width(out) for out in _publication_destinations(op))
    ):
        _fail(
            "register.publish",
            "publication must cover the source and output widths exactly",
            op,
        )
    result: list[tuple[Value, int]] = []
    for location in locations:
        if (
            not isinstance(location, tuple)
            or len(location) != 2
            or type(location[0]) is not int
            or not 0 <= location[0] < len(sources)
            or type(location[1]) is not int
            or not 0 <= location[1] < register_width(sources[location[0]])
        ):
            _fail("register.publish", "publication locations must be in bounds", op)
        result.append((sources[location[0]], location[1]))
    if len(set(result)) != len(result):
        _fail("register.publish", "publication locations must be distinct", op)
    return tuple(result)


def semantic_hidden_width(program: HybridProgram) -> int:
    publication = _publication(program)
    published = () if publication is None else _publication_sources(publication)
    return sum(
        register_width(op.results[0])
        for op in program.body.blocks[0].operations
        if op.name == "qreg.hidden"
        and not op.attributes.get("workspace", False)
        and op.results[0] not in published
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
    publication = _publication(program)
    if publication is not None:
        owners = {
            value: [-1] * register_width(value)
            for value in _publication_sources(publication)
        }
        outputs = _publication_destinations(publication)
        locations = _publication_layout(publication)
        target_wires = (
            wire for value in outputs for wire in reversed(values[value.id])
        )
        for (owner, bit), wire in zip(locations, target_wires, strict=True):
            owners[owner][-1 - bit] = wire
        values.update((owner.id, tuple(ports)) for owner, ports in owners.items())
    following = position
    for op in program.body.blocks[0].operations:
        if op.name == "qreg.hidden" and op.results[0].id not in values:
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


def _borrowed_program(
    operations: Sequence[Operation],
    readonly: Sequence[Value],
    writable: Sequence[Value],
    *,
    name: str = "$borrow",
) -> HybridProgram:
    """Model initialized borrowed storage without assuming actual zero entry.

    This envelope models incoming writable storage as already initialized.
    Its allocation operations prove
    initialization structurally; they make no zero-state assumption about the
    actual caller storage or the captured unitary effects.
    """
    scope = name
    occupied = {value.id.scope for value in (*readonly, *writable)}
    occupied.update(
        value.id.scope
        for operation in operations
        for value in (*operation.operands, *operation.results)
    )
    while scope in occupied:
        scope += "$"
    entry = Value(ValueId(scope, 0), QUANTUM_EFFECT)
    effect = entry
    counter = 1
    checked: list[Operation] = []
    for index, value in enumerate(writable):
        checked.append(
            Operation(
                "qreg.declare", results=(value,), attributes={"name": f"$inout{index}"}
            )
        )
        following = Value(ValueId(scope, counter), QUANTUM_EFFECT)
        counter += 1
        checked.append(Operation("qreg.allocate", (effect, value), (following,)))
        effect = following
    for operation in operations:
        if not operation.operands:
            checked.append(operation)
            continue
        following = Value(ValueId(scope, counter), QUANTUM_EFFECT)
        counter += 1
        checked.append(
            replace(
                operation,
                operands=(effect, *operation.operands[1:]),
                results=(following,),
            )
        )
        effect = following
    checked.append(Operation("qreg.return", (effect, *readonly, *writable)))
    return HybridProgram(name, Region((Block((entry, *readonly), tuple(checked)),)))


def _verify_borrowed_effects(
    operations: Sequence[Operation],
    functions: Sequence[HybridProgram],
    readonly: Sequence[Value],
    writable: Sequence[Value],
) -> None:
    """Check borrowed effects with the existing Const and lifecycle verifier."""
    from .register_verifier import _verify_body

    envelope = _borrowed_program(operations, readonly, writable)
    arithmetic = tuple(f for f in functions if not is_register_function(f))
    helpers = {f.name: f for f in functions if is_register_function(f)}
    _verify_body(envelope, arithmetic, helpers, 256, const_entry=True)


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
    from .register_verifier import _verify_body

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
                if op.name in {
                    "qreg.call",
                    "qreg.control_call",
                    "qreg.replay",
                    "qreg.unitary_call",
                }:
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
        _verify_body(expanded, arithmetic, helpers, max_qubits)
    return program


def verify_operation_attributes(op: Operation) -> None:
    """Check operation attributes before inspecting stateful register use."""
    attributes = _ATTRIBUTES.get(op.name)
    if (
        op.name == "qreg.gate"
        and isinstance(op.attributes.get("gate"), str)
        and (op.attributes["gate"] in _ANGLE_GATES)
    ):
        assert attributes is not None
        attributes = attributes | {"angle"}
    if (
        op.name in {"qreg.gate", "qreg.control_qft", "qreg.inplace_prepare_amplitudes"}
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
            isinstance((gate := op.attributes.get("gate")), str)
            and gate in _ANGLED_PAIR_GATES
        ):
            attributes = attributes | {"angle"}
        if "indices" in op.attributes:
            attributes = attributes | {"indices"}
    if (
        op.name
        in {"qreg.control_x", "qreg.control_phase", "qreg.inplace_prepare_amplitudes"}
        and "indices" in op.attributes
    ):
        assert attributes is not None
        attributes = attributes | {"indices"}
    if (
        "control_state" in op.attributes
        and "indices" in op.attributes
        and (
            op.name
            in {
                "qreg.control_x",
                "qreg.control_phase",
                "qreg.inplace_prepare_amplitudes",
            }
        )
    ):
        state = op.attributes["control_state"]
        operand_count = len(op.operands) - 1
        count = (
            operand_count - 1
            if op.name == "qreg.control_x"
            else (
                operand_count
                if op.name == "qreg.control_phase"
                else op.attributes.get("control_count")
            )
        )
        if (
            type(count) is not int
            or not 1 <= count <= 8
            or type(state) is not int
            or (not 0 <= state < 1 << count)
        ):
            _fail("register.control", "control state must fit its indexed controls", op)
        assert attributes is not None
        attributes = attributes | {"control_state"}
    if (
        op.name == "qreg.inplace_prepare_amplitudes"
        and "control_count" in op.attributes
    ):
        assert attributes is not None
        attributes = attributes | {"control_count"}
    if op.name == "qreg.declare" and (
        "fields" in op.attributes or "record_type" in op.attributes
    ):
        assert attributes is not None
        attributes = attributes | {"fields", "record_type"}
    if op.name in {"qreg.declare", "qreg.hidden"} and "numeric_format" in op.attributes:
        numeric = op.attributes["numeric_format"]
        if (
            len(op.results) != 1
            or not isinstance(numeric, tuple)
            or len(numeric) != 2
            or type(numeric[0]) is not bool
            or type(numeric[1]) is not int
            or not 0 <= numeric[1] <= register_width(op.results[0], op)
            or "fields" in op.attributes
            or "record_type" in op.attributes
        ):
            _fail(
                "register.numeric",
                "numeric format requires a scalar (bool, bounded fractional bits)",
                op,
            )
        assert attributes is not None
        attributes = attributes | {"numeric_format"}
    if op.name == "qreg.hidden" and "workspace" in op.attributes:
        if op.attributes["workspace"] is not True:
            _fail(
                "register.workspace", "hidden workspace marker must be exactly True", op
            )
        assert attributes is not None
        attributes = attributes | {"workspace"}
    if op.name == "qreg.unitary":
        assert attributes is not None
        attributes = attributes | op.attributes.keys() & {
            "indices",
            "control_index",
            "power",
        }
    if op.name == "qreg.phase" and "angle" in op.attributes:
        angle = op.attributes["angle"]
        if type(angle) is not float or not math.isfinite(angle):
            _fail("register.phase", "phase angle must be a finite float", op)
        attributes = (
            frozenset({"angle", "callee"})
            if len(op.operands) > 1
            else frozenset({"angle"})
        )
    if op.name in {"qreg.call", "qreg.unitary_call"} and "inverse" in op.attributes:
        if op.attributes["inverse"] is not True:
            _fail("register.adjoint", "inverse call marker must be true", op)
        attributes = frozenset({"callee", "inverse"})
    if attributes is None or set(op.attributes) != attributes or op.regions:
        _fail("register.operation", "unsupported register operation or attributes", op)
