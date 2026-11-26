# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register placement follows Qualtran v0.7.0 cirq_optree_to_cbloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_cirq_to_bloq.py

"""Import a supported Cirq operation tree into the typed register Program."""

from collections.abc import Hashable, Iterable, Iterator, Mapping
from math import prod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.primitives.global_phase import (
    scalar_phase_ir,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from ._channel import _channel_instruction
from ._feedback_import import (
    _has_explicit_matrix,
    _lower_feedback_gate,
    _matrix_instruction,
    _scalar_control_operations,
    import_cirq_feedback,
)
from ._gate_metadata import cirq_gate_signature
from ._qubit_manager import InteropQubitManager
from ._register_definition import _expand_native_definitions, _native_definition_module
from ._register_export import _register_shape
from ._source_graph_operations import source_graph_operations
from .conversion import (
    _check_cirq_symbol_identity,
    _cirq,
    _eigen_phase_operations,
    _global_phase_angle,
    _raise_if_lossy,
    _report,
)
from .models import CirqConversionIssue

Qureg = NDArray[np.object_]


class _CleanDecompositionManager(InteropQubitManager):
    def qborrow(self, n: int, dim: int = 2) -> list[Any]:
        raise ValueError("Cirq register import requires clean decomposition workspace")


def _register_operations(
    circuit: Any,
    declared: Mapping[Hashable, FunctionRecord],
    *,
    native_gates: Iterator[Any] | None = None,
) -> Iterator[Any]:
    """Resolve SDK bodies while reserving every live sibling operand."""
    cirq = _cirq()
    manager = _CleanDecompositionManager(
        cirq.GreedyQubitManager(prefix="register_work", maximize_reuse=True)
    )
    context = cirq.DecompositionContext(manager)

    def expand(
        operations: Iterable[Any],
        ancestors: tuple[Any, ...],
        *,
        allow_definitions: bool = True,
    ) -> Iterator[Any]:
        original = tuple(operations)
        _check_cirq_symbol_identity(cirq, original)
        frame = []
        for operation in original:
            gate = operation.without_classical_controls().gate
            definition = (
                declared.get(gate)
                if allow_definitions and isinstance(gate, Hashable)
                else None
            )
            if allow_definitions and (
                definition is not None
                or getattr(gate, "flagquantum_register_definition", None) is not None
            ):
                frame.append(operation)
            else:
                frame.extend(
                    _scalar_control_operations(
                        cirq, _eigen_phase_operations(cirq, (operation,))
                    )
                )
        with manager.reserve_qubits(q for op in frame for q in op.qubits):
            for operation in frame:
                if operation.tags:
                    raise ValueError(
                        "Cirq register import requires untagged gate operations"
                    )
                if operation.classical_controls:
                    for leaf in expand(
                        (operation.without_classical_controls(),),
                        ancestors,
                        allow_definitions=allow_definitions,
                    ):
                        yield leaf.with_classical_controls(
                            *operation.classical_controls
                        )
                    continue
                if not isinstance(operation.gate, cirq.Gate):
                    raise ValueError(
                        "Cirq register import requires untagged gate operations"
                    )
                gate = operation.gate
                if any(qubit.dimension != 2 for qubit in operation.qubits):
                    raise ValueError(
                        "Cirq register import requires two-dimensional operands"
                    )
                definition = (
                    declared.get(gate)
                    if allow_definitions and isinstance(gate, Hashable)
                    else None
                )
                has_definition = allow_definitions and (
                    definition is not None
                    or getattr(gate, "flagquantum_register_definition", None)
                    is not None
                )
                if has_definition and native_gates is not None:
                    yield from expand(
                        (next(native_gates).on(*operation.qubits),),
                        ancestors,
                        allow_definitions=False,
                    )
                    continue
                if (
                    has_definition
                    or isinstance(gate, (cirq.GlobalPhaseGate, cirq.ResetChannel))
                    or _lower_feedback_gate(cirq, gate, [], 0) is not None
                    or _has_explicit_matrix(cirq, gate)
                ):
                    yield operation
                    continue
                if gate in ancestors or len(ancestors) >= 256:
                    raise ValueError(
                        "Cirq register decomposition is cyclic or too deep"
                    )
                tree = source_graph_operations(operation, manager)
                if tree is None:
                    tree = cirq.decompose_once(
                        operation, context=context, flatten=False, default=None
                    )
                if tree is None:
                    # Match the SDK wrapper's tensor-backed unitary fallback only
                    # after its declared decomposition has declined the operation.
                    source_leaf = getattr(gate, "bloq", gate)
                    while getattr(source_leaf, "ctrl_spec", None) is not None:
                        source_leaf = source_leaf.subbloq
                    unitary_hook = getattr(source_leaf, "_has_unitary_", None)
                    if (
                        hasattr(source_leaf, "signature")
                        and callable(unitary_hook)
                        and unitary_hook() is False
                    ):
                        raise ValueError(
                            "Source leaf explicitly declares no unitary and has "
                            f"no executable decomposition: {gate}"
                        )
                    matrix = cirq.unitary(gate, default=None)
                    if matrix is None:
                        yield operation
                    else:
                        yield cirq.MatrixGate(matrix).on(*operation.qubits)
                else:
                    yield from expand(
                        cirq.flatten_to_ops(tree),
                        (*ancestors, gate),
                        allow_definitions=allow_definitions,
                    )

    yield from expand(circuit.all_operations(), ())


def _port_arrays(
    registers: Iterable[Register], quregs: Mapping[str, object]
) -> dict[str, Qureg]:
    cirq = _cirq()
    ports = tuple(registers)
    if {reg.name for reg in ports} != set(quregs):
        raise ValueError("Cirq qureg names must match the register signature")
    seen: set[Hashable] = set()
    arrays: dict[str, Qureg] = {}
    for reg in ports:
        shape = _register_shape(reg)
        supplied = quregs[reg.name]
        values = np.asarray(
            [supplied] if isinstance(supplied, cirq.Qid) else supplied, dtype=object
        )
        if values.shape != shape:
            raise ValueError(f"Cirq register {reg.name!r} requires qureg shape {shape}")
        for qubit in values.flat:
            if not isinstance(qubit, cirq.Qid) or qubit.dimension != 2 or qubit in seen:
                raise ValueError("Cirq ports require distinct two-dimensional qubits")
            seen.add(qubit)
        arrays[reg.name] = values
    return arrays


def _operation_ports(
    signature: Signature, qubits: tuple[Hashable, ...]
) -> tuple[dict[str, Qureg], dict[str, Qureg]]:
    incoming: dict[str, Qureg] = {}
    outgoing: dict[str, Qureg] = {}
    offset = 0
    for register in signature:
        shape = _register_shape(register)
        width = prod(shape)
        wires = qubits[offset : offset + width]
        if len(wires) != width:
            raise ValueError("Cirq gate operands do not match the declared signature")
        placed = np.asarray(wires, dtype=object).reshape(shape)
        if register.side & RegisterSide.LEFT:
            incoming[register.name] = placed
        if register.side & RegisterSide.RIGHT:
            outgoing[register.name] = placed
        offset += width
    if offset != len(qubits):
        raise ValueError("Cirq gate operands do not match the declared signature")
    return incoming, outgoing


class _WordBindings:
    """Track live word groups while SDK qubits still have boundary identity."""

    def __init__(self, builder: ComponentProgramBuilder) -> None:
        self.builder = builder
        self.words: dict[tuple[Hashable, ...], RegisterBinding] = {}
        self.empty_words: dict[tuple[str, tuple[int, ...]], RegisterBinding] = {}
        self.bit = quantum_bit_type()

    def store(
        self,
        registers: Iterable[Register],
        ports: Mapping[str, Qureg],
        bindings: Mapping[str, object],
    ) -> None:
        for register in registers:
            words = np.asarray(bindings[register.name], dtype=object)
            for index in register.all_idxs():
                qubits = tuple(ports[register.name][index])
                if not qubits:
                    key = (register.name, index)
                    if key in self.empty_words:
                        raise ValueError("Cirq output overlaps a live empty register")
                    self.empty_words[key] = words[index]
                    continue
                if any(set(qubits).intersection(group) for group in self.words):
                    raise ValueError("Cirq gate output overlaps a live register")
                self.words[qubits] = words[index]

    def take(
        self,
        qubits: tuple[Hashable, ...],
        dtype: RegisterType,
        empty_key: tuple[str, tuple[int, ...]],
    ) -> RegisterBinding:
        if not qubits:
            word = self.empty_words.pop(empty_key, None)
            if word is None:
                return self.builder.allocate(dtype=dtype)
            if word.register.dtype == dtype:
                return word
            return self.builder.join(self.builder.split(word), dtype)

        direct = self.words.get(qubits)
        if direct is not None and direct.register.dtype == dtype:
            return self.words.pop(qubits)

        requested = set(qubits)
        overlaps = [group for group in self.words if requested.intersection(group)]
        if not overlaps:
            return self.builder.allocate(dtype=dtype)

        for group in overlaps:
            word = self.words[group]
            if len(group) != 1 or word.register.dtype != self.bit:
                del self.words[group]
                split_bits = self.builder.split(word)
                self.words.update(
                    ((qubit,), bit)
                    for qubit, bit in zip(group, split_bits, strict=True)
                )

        bits = [
            (
                self.words.pop((qubit,))
                if (qubit,) in self.words
                else self.builder.allocate(dtype=self.bit)
            )
            for qubit in qubits
        ]
        return bits[0] if dtype == self.bit else self.builder.join(bits, dtype)

    def take_ports(
        self, registers: Iterable[Register], ports: Mapping[str, Qureg]
    ) -> dict[str, object]:
        bindings: dict[str, object] = {}
        for register in registers:
            words = np.empty(register.shape, dtype=object)
            for index in register.all_idxs():
                words[index] = self.take(
                    tuple(ports[register.name][index]),
                    register.dtype,
                    (register.name, index),
                )
            bindings[register.name] = words if register.shape else words[()]
        return bindings


def import_cirq_register_optree(
    optree: object,
    *,
    signature: Signature | None = None,
    in_quregs: Mapping[str, object] | None = None,
    out_quregs: Mapping[str, object] | None = None,
    gate_definitions: Mapping[Hashable, FunctionRecord] | None = None,
) -> ProgramRecord:
    """Import admitted gates, measurement and feedback with explicit placement.

    Qureg arrays use the register shape followed by the ordered bit axis.
    Without a signature, sorted circuit qubits form one through register.
    Omitted input wires are allocated at first use. Wires outside the output
    mapping are released under the clean-work contract, which callers must
    satisfy when binding and executing the Program. Gate definitions can
    provide native typed operand groups for selected SDK gates. A gate can
    supply its own FunctionRecord as `flagquantum_register_definition`.
    An explicit definition takes precedence over recursive SDK decomposition.
    Recursive decompositions require clean workspace and a maximum nesting depth
    of 256. Explicit unitary leaves become owned matrix operations. A tensor-only
    unitary fallback is consulted after the SDK decomposition declines.
    Measured bodies retain one shared key history across recursive calls.
    Explicit native definitions retain precedence in measured bodies, with
    local measurement keys isolated before joining the shared history.
    """
    cirq = _cirq()
    circuit = cirq.Circuit(optree)
    bit = quantum_bit_type()
    if signature is None:
        if in_quregs is not None or out_quregs is not None:
            raise ValueError("Cirq qureg placement requires a signature")
        circuit_qubits = sorted(circuit.all_qubits())
        signature = Signature((Register("qubits", bit, shape=(len(circuit_qubits),)),))
        in_quregs = out_quregs = {
            "qubits": np.asarray(circuit_qubits, dtype=object).reshape(
                len(circuit_qubits), 1
            )
        }
    elif in_quregs is None or out_quregs is None:
        raise ValueError("Cirq signature placement requires input and output quregs")

    incoming = _port_arrays(signature.lefts(), in_quregs)
    outgoing = _port_arrays(signature.rights(), out_quregs)
    declared = {} if gate_definitions is None else gate_definitions
    definitions: dict[str, FunctionRecord] = {}
    gates: list[tuple[object, tuple[Hashable, ...], FunctionRecord | None]] = []
    operations = tuple(_register_operations(circuit, declared))
    _check_cirq_symbol_identity(cirq, operations)
    if any(op.classical_controls or cirq.is_measurement(op) for op in operations):
        replacements = _expand_native_definitions(operations, declared)
        native_gates = tuple(
            replacement.without_classical_controls().gate
            for original, replacement in zip(operations, replacements, strict=True)
            if original is not replacement
        )
        # Replay the original tree in one allocation context. Re-expanding the
        # already flattened SDK work would reserve it throughout the whole body
        # and prevent native calls from reusing a released source work wire.
        if native_gates:
            operations = tuple(
                _register_operations(circuit, declared, native_gates=iter(native_gates))
            )
        # Lower one history so feedback across nested SDK bodies and repeated
        # keys retains its SSA producers. Individual per-gate calls lose them.
        qubits = tuple(sorted({q for op in operations for q in op.qubits}))
        feedback = import_cirq_feedback(
            cirq.Circuit(cirq.Moment((op,)) for op in operations), qubit_order=qubits
        )
        registers = tuple(Register(f"bit{i}", bit) for i in range(len(qubits)))
        function = bit_gate_function("feedback", registers, feedback.ir)
        definitions[function.name] = function
        gates.append((None, qubits, function))
        operations = ()
    for operation in operations:
        if not isinstance(operation.gate, cirq.Gate) or operation.tags:
            raise ValueError("Cirq register import requires untagged gate operations")
        gate = operation.gate
        qubits = operation.qubits
        if any(qubit.dimension != 2 for qubit in qubits):
            raise ValueError("Cirq register import requires two-dimensional operands")
        definition = (
            declared.get(gate) if declared and isinstance(gate, Hashable) else None
        )
        if definition is None:
            definition = getattr(gate, "flagquantum_register_definition", None)
        if definition is not None:
            if (
                not isinstance(definition, FunctionRecord)
                or definition.registers is None
            ):
                raise TypeError(
                    "Cirq gate definition requires a typed native FunctionRecord"
                )
            previous = definitions.setdefault(definition.name, definition)
            if previous != definition:
                raise ValueError(
                    f"Conflicting Cirq gate definition {definition.name!r}"
                )
        gates.append((gate, qubits, definition))

    definitions = _native_definition_module(definitions.values(), declared)
    builder = ComponentProgramBuilder(signature)
    live = _WordBindings(builder)
    live.store(signature.lefts(), incoming, builder.inputs)

    def call(function: FunctionRecord, qubits: tuple[Hashable, ...]) -> None:
        assert function.registers is not None
        ports = Signature(function.registers)
        left, right = _operation_ports(ports, qubits)
        values = live.take_ports(ports.lefts(), left)
        if function.name in definitions:
            outputs = builder.call_program(
                ProgramRecord(tuple(definitions.values())), values, entry=function.name
            )
        else:
            outputs = builder.call_dict(function.name, values)
        live.store(ports.rights(), right, outputs)

    for index, (gate, qubits, definition) in enumerate(gates):
        if definition is not None:
            call(definition, qubits)
            continue
        if isinstance(gate, cirq.GlobalPhaseGate):
            # A phase-only program allocates a clean scalar carrier from the pool.
            qubits = (next(iter(live.words))[0] if live.words else object(),)
            instructions = scalar_phase_ir(
                0, _global_phase_angle(cirq, gate.coefficient)
            )
        elif isinstance(gate, cirq.ResetChannel) and len(qubits) == 1:
            instructions = (Instruction("reset", (0,), metadata={"is_dynamic": True}),)
        elif _has_explicit_matrix(cirq, gate):
            instructions = (_matrix_instruction(cirq, gate, tuple(range(len(qubits)))),)
        elif (
            channel := _channel_instruction(gate, tuple(range(len(qubits))))
        ) is not None:
            instructions = (channel,)
        else:
            if not qubits:
                raise ValueError(
                    "Cirq register import requires nonempty two-dimensional operands"
                )
            issues: list[CirqConversionIssue] = []
            lowered = _lower_feedback_gate(cirq, gate, issues, index)
            _raise_if_lossy(
                _report("from_cirq", str(cirq.__version__), issues), allow_lossy=False
            )
            assert lowered is not None
            opcode, params = lowered
            instructions = (
                Instruction(opcode, tuple(range(len(qubits))), params=params),
            )
        names = tuple(f"bit{i}" for i in range(len(qubits)))
        registers = tuple(Register(name, bit) for name in names)
        name = f"gate_{index}"
        while name in definitions:
            name = "_" + name
        function = bit_gate_function(
            name, registers, CircuitIR(len(qubits), instructions)
        )
        builder.define(function)
        call(function, qubits)

    returns = live.take_ports(signature.rights(), outgoing)
    for word in (*live.words.values(), *live.empty_words.values()):
        builder.free(word)
    return builder.finalize(returns)


def import_cirq_register_gate(
    value: object,
    *,
    signature: Signature | None = None,
    in_quregs: Mapping[str, object] | None = None,
    out_quregs: Mapping[str, object] | None = None,
    gate_definitions: Mapping[Hashable, FunctionRecord] | None = None,
) -> ProgramRecord | None:
    """Import one declared SDK decomposition with explicit native ownership.

    The raw gate contract exposes the original operands as through wires.
    A supplied signature can express sided ports and defaults to operand
    order. Temporary wires use the optree importer's clean lifetimes.
    """
    cirq = _cirq()
    if isinstance(value, cirq.Gate):
        value = value.on(*cirq.LineQid.for_gate(value))
    if (
        not isinstance(value, cirq.Operation)
        or not isinstance(value.gate, cirq.Gate)
        or value.tags
    ):
        raise ValueError(
            "Cirq register decomposition requires an untagged gate operation"
        )
    if signature is None:
        if in_quregs is not None or out_quregs is not None:
            raise ValueError("Cirq qureg placement requires a signature")
        signature = cirq_gate_signature(value.gate)
    if in_quregs is None and out_quregs is None:
        in_quregs, out_quregs = _operation_ports(signature, value.qubits)
    elif in_quregs is None or out_quregs is None:
        raise ValueError("Cirq signature placement requires input and output quregs")
    manager = _CleanDecompositionManager(
        cirq.GreedyQubitManager(prefix="register_work", maximize_reuse=True)
    )
    manager.manage_qubits(value.qubits)
    operations = cirq.decompose_once(
        value, context=cirq.DecompositionContext(manager), default=None
    )
    if operations is None:
        return None
    return import_cirq_register_optree(
        cirq.Circuit(cirq.Moment((operation,)) for operation in operations),
        signature=signature,
        in_quregs=in_quregs,
        out_quregs=out_quregs,
        gate_definitions=gate_definitions,
    )
