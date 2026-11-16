# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register naming follows Qualtran v0.7.0 get_named_qubits.

"""Bind concrete FlagQuantum register ports to external Cirq qubits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ...core.ir import ensure_circuit_ir
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .conversion import _cirq, export_cirq
from .models import CirqConversionReport, CirqExportResult

_Qureg = NDArray[np.object_]


def _register_shape(register: Register) -> tuple[int, ...]:
    width = register.dtype.num_qubits
    shape = register.shape_symbolic
    if (
        type(width) is not int
        or width < 0
        or register.dtype.num_cbits != 0
        or any(type(extent) is not int or extent < 0 for extent in shape)
    ):
        raise ValueError(
            f"Cirq placement requires a concrete quantum register: {register.name}"
        )
    return (*register.shape, width)


def _named_qubits(cirq: Any, register: Register, shape: tuple[int, ...]) -> _Qureg:
    width = shape[-1]
    if len(shape) == 1:
        qubits = (
            [cirq.NamedQubit(register.name)]
            if width == 1
            else cirq.NamedQubit.range(width, prefix=register.name)
        )
        return np.array(qubits, dtype=object)
    qubits = np.empty(shape, dtype=object)
    for element in register.all_idxs():
        prefix = f"{register.name}[{', '.join(map(str, element))}]"
        for bit in range(width):
            suffix = "" if width == 1 else f"[{bit}]"
            qubits[element + (bit,)] = cirq.NamedQubit(prefix + suffix)
    return qubits


def export_register_cirq(
    program: Any,
    signature: Signature,
    wire_ports: Mapping[str, Sequence[int]],
    *,
    input_quregs: Mapping[str, Any] | None = None,
    allow_lossy: bool = False,
) -> tuple[CirqExportResult, dict[str, _Qureg]]:
    """Place a static export on named register wires and return right ports.

    The native circuit owns its physical allocation. Unbound native wires
    remain temporary lines, and right-only ports have fresh generated names.
    This boundary preserves the ordinary export diagnostics and instruction
    order. A caller-supplied qubit array must match the register's shape with
    its bit axis last.
    """
    ir = ensure_circuit_ir(program)
    names = tuple(reg.name for reg in signature)
    left_names = {reg.name for reg in signature.lefts()}
    if len(names) != len(set(names)) or set(wire_ports) != set(names):
        raise ValueError(
            "Cirq placement requires one native wire binding per unique register"
        )
    if input_quregs is not None and set(input_quregs) != left_names:
        raise ValueError("Cirq input placement must bind exactly the left registers")

    cirq = _cirq()
    placed: dict[int, Any] = {}
    occupied: set[Any] = set()
    all_ports: dict[str, _Qureg] = {}
    # Reserve external inputs first so generated outputs cannot capture an input.
    registers = tuple(signature.lefts()) + tuple(
        reg for reg in signature.rights() if reg.name not in left_names
    )
    for register in registers:
        name = register.name
        shape = _register_shape(register)
        supplied_port = input_quregs is not None and name in left_names
        if input_quregs is not None and supplied_port:
            supplied = input_quregs[name]
            qubits = np.array(
                [supplied] if isinstance(supplied, cirq.Qid) else supplied, dtype=object
            )
        else:
            qubits = _named_qubits(cirq, register, shape)
        if qubits.shape != shape:
            raise ValueError(
                f"Cirq register {name!r} requires qubit array shape {shape}"
            )
        wires = wire_ports[name]
        if len(wires) != qubits.size:
            raise ValueError(
                f"Cirq register {name!r} requires {qubits.size} native wires"
            )
        for index, (wire, qubit) in enumerate(zip(wires, qubits.flat, strict=True)):
            if type(wire) is not int or not 0 <= wire < ir.n_wires or wire in placed:
                raise ValueError(
                    "Cirq placement requires distinct in-range native wires"
                )
            if not isinstance(qubit, cirq.Qid) or qubit.dimension != 2:
                raise ValueError("Cirq placement requires two-dimensional qubits")
            if supplied_port and qubit in occupied:
                raise ValueError("Cirq placement requires distinct qubits")
            while qubit in occupied:
                qubit = cirq.NamedQubit(f"{qubit}_")
            qubits.flat[index] = qubit
            placed[wire] = qubit
            occupied.add(qubit)
        all_ports[name] = qubits

    for wire in range(ir.n_wires):
        if wire in placed:
            continue
        qubit = cirq.LineQubit(wire)
        name = f"__fq_work_{wire}"
        while qubit in occupied:
            qubit = cirq.NamedQubit(name)
            name += "_"
        placed[wire] = qubit
        occupied.add(qubit)

    exported = export_cirq(ir, allow_lossy=allow_lossy)
    circuit = exported.circuit.transform_qubits(lambda qubit: placed[qubit.x])
    return CirqExportResult(circuit, cast(CirqConversionReport, exported.report)), {
        reg.name: all_ports[reg.name] for reg in signature.rights()
    }
