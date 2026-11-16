# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register placement follows Qualtran v0.7.0 cirq_optree_to_cbloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_cirq_to_bloq.py

"""Import a supported Cirq operation tree into the typed register Program."""

from collections.abc import Hashable, Iterable, Mapping
from math import prod

import numpy as np
from numpy.typing import NDArray

from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.primitives.global_phase import (
    global_phase_exponent,
    global_phase_ir,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from ._decompose import decompose_cirq_once
from ._gate_metadata import cirq_gate_signature
from ._register_export import _register_shape
from .conversion import _cirq, _lower_gate, _raise_if_lossy, _report
from .models import CirqConversionIssue

Qureg = NDArray[np.object_]


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
        if shape[-1] == 0:
            raise ValueError("Cirq register import requires positive word widths")
        values = np.asarray(quregs[reg.name], dtype=object)
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
        if shape[-1] == 0:
            raise ValueError("Cirq register import requires positive word widths")
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
                if any(set(qubits).intersection(group) for group in self.words):
                    raise ValueError("Cirq gate output overlaps a live register")
                self.words[qubits] = words[index]

    def take(
        self, qubits: tuple[Hashable, ...], dtype: RegisterType
    ) -> RegisterBinding:
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
                    tuple(ports[register.name][index]), register.dtype
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
    """Import admitted gates and single-wire reset with explicit placement.

    Qureg arrays use the register shape followed by the ordered bit axis.
    Without a signature, sorted circuit qubits form one through register.
    Omitted input wires are allocated at first use. Wires outside the output
    mapping are released under the clean-work contract, which callers must
    satisfy when binding and executing the Program. Gate definitions can
    provide native typed operand groups for selected SDK gates.
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
    builder = ComponentProgramBuilder(signature)
    live = _WordBindings(builder)
    live.store(signature.lefts(), incoming, builder.inputs)
    definitions: dict[str, FunctionRecord] = {}
    declared = {} if gate_definitions is None else gate_definitions
    reserved = {function.name for function in declared.values()}

    def call(function: FunctionRecord, qubits: tuple[Hashable, ...]) -> None:
        if function.registers is None:
            raise ValueError("Cirq gate definition requires typed register ports")
        previous = definitions.get(function.name)
        if previous is None:
            builder.define(function)
            definitions[function.name] = function
        elif previous != function:
            raise ValueError(f"Conflicting Cirq gate definition {function.name!r}")
        ports = Signature(function.registers)
        left, right = _operation_ports(ports, qubits)
        outputs = builder.call_dict(function.name, live.take_ports(ports.lefts(), left))
        live.store(ports.rights(), right, outputs)

    for index, operation in enumerate(circuit.all_operations()):
        if not isinstance(operation, cirq.GateOperation) or operation.tags:
            raise ValueError("Cirq register import requires untagged gate operations")
        gate = operation.gate
        qubits = operation.qubits
        definition = (
            declared.get(gate) if declared and isinstance(gate, Hashable) else None
        )
        if definition is not None:
            if any(qubit.dimension != 2 for qubit in qubits):
                raise ValueError(
                    "Cirq register import requires two-dimensional operands"
                )
            call(definition, qubits)
            continue
        if isinstance(gate, cirq.GlobalPhaseGate):
            if cirq.is_parameterized(gate.coefficient):
                raise ValueError("Cirq global phase requires a concrete coefficient")
            # A phase-only program allocates a clean scalar carrier from the pool.
            qubits = (next(iter(live.words))[0] if live.words else object(),)
            exponent = global_phase_exponent(complex(gate.coefficient))
            instructions = global_phase_ir(float(exponent), anchor=0)
        elif isinstance(gate, cirq.ISwapPowGate) and gate in (
            cirq.ISWAP,
            cirq.ISWAP**-1,
        ):
            opcode = "iswap" if gate == cirq.ISWAP else "iswapdg"
            instructions = (Instruction(opcode, (0, 1)),)
        elif (
            isinstance(gate, cirq.ResetChannel)
            and len(qubits) == 1
            and qubits[0].dimension == 2
        ):
            instructions = (Instruction("reset", (0,), metadata={"is_dynamic": True}),)
        else:
            if not qubits or any(qubit.dimension != 2 for qubit in qubits):
                raise ValueError(
                    "Cirq register import requires nonempty two-dimensional operands"
                )
            issues: list[CirqConversionIssue] = []
            lowered = _lower_gate(cirq, gate, issues, index)
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
        while name in reserved:
            name = "_" + name
        function = bit_gate_function(
            name, registers, CircuitIR(len(qubits), instructions)
        )
        call(function, qubits)

    returns = live.take_ports(signature.rights(), outgoing)
    for word in live.words.values():
        builder.free(word)
    return builder.finalize(returns)


def import_cirq_register_gate(
    value: object,
    *,
    signature: Signature | None = None,
    in_quregs: Mapping[str, object] | None = None,
    out_quregs: Mapping[str, object] | None = None,
) -> ProgramRecord | None:
    """Import one declared SDK decomposition with explicit native ownership.

    The raw gate contract exposes the original operands as through wires.
    A supplied signature can express sided ports and defaults to operand order. Temporary decomposition
    wires outside those ports use the optree importer's clean lifetimes.
    """
    cirq = _cirq()
    if isinstance(value, cirq.Gate):
        value = value.on(*cirq.LineQid.for_gate(value))
    if not isinstance(value, cirq.GateOperation) or value.tags:
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
    operations = decompose_cirq_once(value)
    if operations is None:
        return None
    return import_cirq_register_optree(
        cirq.Circuit(cirq.Moment((operation,)) for operation in operations),
        signature=signature,
        in_quregs=in_quregs,
        out_quregs=out_quregs,
    )
