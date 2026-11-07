"""Verify the operations and lifetime of one checked register body."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ...core._composition import _is_finite_unitary
from .._hybrid.model import (
    QUANTUM_EFFECT,
    HybridProgram,
    Operation,
    Value,
    ValueId,
)
from .gate_tape import check_gate_tape
from .pauli_rotation import checked_support
from .profile import _fail, _width
from .register_profile import (
    _ANGLE_GATES,
    _ANGLED_PAIR_GATES,
    _GATES,
    _MAX_INPLACE_AMPLITUDES,
    _PAIR_GATES,
    MAX_MATRIX_QUBITS,
    MAX_PREPARED_AMPLITUDES,
    MAX_TRANSFORM_WIDTH,
    _check_real_preparation,
    _publication,
    _publication_destinations,
    _publication_layout,
    _publication_sources,
    record_fields,
    record_readouts,
    register_ports,
    register_width,
    unitary_matrix,
    verify_operation_attributes,
)


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
    scratch: set[Value] = set()
    publication = _publication(program)
    published = () if publication is None else _publication_destinations(publication)
    for position, op in enumerate(block.operations):
        verify_operation_attributes(op)
        if any(known.get(v.id) != v for v in op.operands):
            _fail("register.use", "undefined or mistyped operand", op)
        if op.name in {"qreg.declare", "qreg.local", "qreg.hidden"}:
            if op.operands or len(op.results) != 1:
                _fail(
                    "register.declare", "declaration requires one register result", op
                )
            value = op.results[0]
            declared_width = register_width(value, op)
            if op.name != "qreg.declare" or value not in published:
                width += declared_width
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
                if op.attributes.get("workspace", False):
                    scratch.add(value)
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
            if op.name == "qreg.publish":
                outputs = _publication_destinations(op)
                sources = _publication_sources(op)
                if (
                    op is not publication
                    or const_entry
                    or local_storage
                    or len(set(outputs)) != len(outputs)
                    or len(set(sources)) != len(sources)
                    or any(v not in returned or v.id in initialized for v in outputs)
                    or any(
                        v not in hidden or v in scratch or v.id not in initialized
                        for v in sources
                    )
                ):
                    _fail(
                        "register.publish",
                        "terminal publication requires fresh outputs and initialized hidden data",
                        op,
                    )
                _publication_layout(op)
                for source in sources:
                    hidden.remove(source)
                    registers.remove(source)
                    initialized.remove(source.id)
                initialized.update(v.id for v in outputs)
            elif op.name in {"qreg.call", "qreg.control_call"}:
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
                    or not 1 <= register_width(arguments[0]) <= 10
                    or not 1 <= register_width(destination) <= 5
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
                control_count = op.attributes.get("control_count", 0)
                if "control_count" in op.attributes and (
                    type(control_count) is not int
                    or not 1 <= control_count <= 7
                    or len(operands) > 8
                    or control_count >= len(operands)
                    or "indices" not in op.attributes
                ):
                    _fail(
                        "register.amplitudes",
                        "invalid indexed preparation control extent",
                        op,
                    )
                targets = operands[control_count:]
                if any(value.id not in initialized for value in operands) or any(
                    value in borrowed for value in targets
                ):
                    _fail(
                        "register.amplitudes",
                        "inplace preparation requires initialized writable storage",
                        op,
                    )
                if "indices" in op.attributes:
                    indices = op.attributes["indices"]
                    if (
                        not isinstance(indices, tuple)
                        or len(indices) != len(operands)
                        or not 1
                        <= len(indices)
                        <= (
                            (
                                _MAX_INPLACE_AMPLITUDES
                                if not control_count
                                else MAX_PREPARED_AMPLITUDES
                            ).bit_length()
                            - 1
                        )
                        or any(
                            type(index) is not int
                            or not 0 <= index < register_width(value)
                            for value, index in zip(operands, indices, strict=True)
                        )
                        or len(set(zip(operands, indices, strict=True))) != len(indices)
                    ):
                        _fail(
                            "register.amplitudes",
                            "preparation indices must be bounded and distinct",
                            op,
                        )
                    width = len(indices) - control_count
                else:
                    if arguments:
                        _fail(
                            "register.amplitudes",
                            "inplace preparation requires one register or an indexed view",
                            op,
                        )
                    width = register_width(destination)
                _check_real_preparation(op, width)
                changed = tuple(dict.fromkeys(targets))
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
            elif op.name == "qreg.gate_tape":
                if (
                    const_entry
                    or len(set(operands)) != len(operands)
                    or operands != tuple(returned)
                    or any(v.id not in initialized for v in operands)
                    or position != len(block.operations) - 2
                    or any(
                        child.name not in {"qreg.declare", "qreg.allocate"}
                        for child in block.operations[:position]
                    )
                ):
                    _fail(
                        "register.gate_tape",
                        "unitary tape requires fresh public output storage",
                        op,
                    )
                check_gate_tape(
                    op, sum(register_width(v) for v in operands), max_qubits
                )
                changed = operands
            elif op.name == "qreg.replay":
                callee, count = (op.attributes[key] for key in ("callee", "count"))
                if (
                    not isinstance(callee, str)
                    or callee not in helpers
                    or type(count) is not int
                    or not 2 <= count <= 256
                ):
                    _fail(
                        "register.replay",
                        "replay requires a known template and bounded count",
                        op,
                    )
                body = helpers[callee].body.blocks[0]
                declared = tuple(
                    child.results[0]
                    for child in body.operations
                    if child.name == "qreg.declare"
                )
                length = len(declared)
                initial = body.operations[length : 2 * length]
                effects = body.operations[2 * length : -1]
                if (
                    len(body.arguments) != 1
                    or tuple(child.name for child in body.operations[:length])
                    != ("qreg.declare",) * length
                    or tuple(child.name for child in initial)
                    != ("qreg.allocate",) * length
                    or tuple(child.operands[1] for child in initial) != declared
                    or not 1 <= len(effects) <= 512
                    or not any(child.name == "qreg.reset" for child in effects)
                    or any(
                        child.name
                        not in {
                            "qreg.gate",
                            "qreg.pair_gate",
                            "qreg.control_x",
                            "qreg.array_phase_flip",
                            "qreg.phase",
                            "qreg.control_phase",
                            "qreg.pauli_rotation",
                            "qreg.qft",
                            "qreg.inverse_qft",
                            "qreg.control_qft",
                            "qreg.inplace_prepare_amplitudes",
                            "qreg.unitary",
                            "qreg.reset",
                        }
                        for child in effects
                    )
                ):
                    _fail(
                        "register.replay",
                        "replay template must contain one checked reset cycle",
                        op,
                    )
                if (
                    tuple(v.type for v in operands) != tuple(v.type for v in declared)
                    or len(set(operands)) != len(operands)
                    or any(v in borrowed or v.id not in initialized for v in operands)
                ):
                    _fail(
                        "register.replay",
                        "replay requires disjoint initialized writable operands",
                        op,
                    )
                changed = operands
            elif op.name == "qreg.reset":
                bit = op.attributes["index"]
                if (
                    arguments
                    or destination in borrowed
                    or destination.id not in initialized
                    or type(bit) is not int
                    or not 0 <= bit < register_width(destination)
                ):
                    _fail(
                        "register.reset",
                        "reset needs a writable initialized scalar bit",
                        op,
                    )
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
                        _check_real_preparation(op, register_width(destination))
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
