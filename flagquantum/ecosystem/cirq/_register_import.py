# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register placement follows Qualtran v0.7.0 cirq_optree_to_cbloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_cirq_to_bloq.py

"""Import a supported Cirq operation tree into the typed register Program."""

from collections.abc import Hashable, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.primitives.global_phase import (
    global_phase_exponent,
    global_phase_ir,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ._decompose import decompose_cirq_once
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


def import_cirq_register_optree(
    optree: object,
    *,
    signature: Signature | None = None,
    in_quregs: Mapping[str, object] | None = None,
    out_quregs: Mapping[str, object] | None = None,
) -> ProgramRecord:
    """Import admitted gates and single-wire reset with explicit placement.

    Qureg arrays use the register shape followed by the ordered bit axis.
    Without a signature, sorted circuit qubits form one through register.
    Omitted input wires are allocated at first use. Wires outside the output
    mapping are released under the clean-work contract, which callers must
    satisfy when binding and executing the Program.
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
    live: dict[Hashable, RegisterBinding] = {}

    for reg in signature.lefts():
        words = np.asarray(builder.inputs[reg.name], dtype=object)
        for idx in reg.all_idxs():
            word = words[idx]
            bits = (word,) if reg.dtype == bit else builder.split(word)
            live.update(zip(incoming[reg.name][idx], bits, strict=True))

    def ensure(qubit: Hashable) -> RegisterBinding:
        if qubit not in live:
            live[qubit] = builder.allocate(dtype=bit)
        return live[qubit]

    for index, operation in enumerate(circuit.all_operations()):
        if not isinstance(operation, cirq.GateOperation) or operation.tags:
            raise ValueError("Cirq register import requires untagged gate operations")
        gate = operation.gate
        qubits = operation.qubits
        if isinstance(gate, cirq.GlobalPhaseGate):
            if cirq.is_parameterized(gate.coefficient):
                raise ValueError("Cirq global phase requires a concrete coefficient")
            # A phase-only program allocates a clean scalar carrier from the pool.
            qubits = (next(iter(live)) if live else object(),)
            exponent = global_phase_exponent(complex(gate.coefficient))
            instructions = global_phase_ir(float(exponent), anchor=0)
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
        function = bit_gate_function(
            f"gate_{index}", registers, CircuitIR(len(qubits), instructions)
        )
        builder.define(function)
        outputs = builder.call_tuple(
            function.name,
            {name: ensure(qubit) for name, qubit in zip(names, qubits, strict=True)},
        )
        for qubit, word in zip(qubits, outputs, strict=True):
            assert isinstance(word, RegisterBinding)
            live[qubit] = word

    returns: dict[str, object] = {}
    for reg in signature.rights():
        words = np.empty(reg.shape, dtype=object)
        for idx in reg.all_idxs():
            qubits = outgoing[reg.name][idx]
            output_bits = [ensure(qubit) for qubit in qubits]
            for qubit in qubits:
                del live[qubit]
            words[idx] = (
                output_bits[0]
                if reg.dtype == bit
                else builder.join(output_bits, reg.dtype)
            )
        returns[reg.name] = words if reg.shape else words[()]
    for word in live.values():
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
    A supplied signature can express sided ports. Temporary decomposition
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
        width = len(value.qubits)
        shape = (width,) if width > 1 else ()
        ports = () if width == 0 else (Register("q", quantum_bit_type(), shape=shape),)
        signature = Signature(ports)
        in_quregs = out_quregs = (
            {"q": np.asarray(value.qubits, dtype=object).reshape((*shape, 1))}
            if width
            else {}
        )
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
