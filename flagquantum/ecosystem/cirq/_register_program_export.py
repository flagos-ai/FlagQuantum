# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Program export follows Qualtran v0.7.0 _cbloq_to_cirq_circuit.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_bloq_to_cirq.py

"""Export a compiler-bound Program while retaining its output qubit leases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ...compiler.registers.register_binding import BoundRegisterProgram
from ...core.ir.program.signatures import Signature
from ._register_export import _named_qubits, _register_shape
from ._register_workspace import _replay_workspace_operations
from .conversion import _cirq, _export_cirq
from .models import CirqConversionReport, CirqExportResult

if TYPE_CHECKING:
    from cirq import Qid, QubitManager


def export_bound_register_cirq(
    bound: BoundRegisterProgram,
    signature: Signature,
    *,
    input_quregs: Mapping[str, Any] | None = None,
    qubit_manager: QubitManager | None = None,
    feedback_key_prefix: str | None = None,
) -> tuple[CirqExportResult, dict[str, NDArray[np.object_]]]:
    """Export a Program bound with consecutive input wires starting at zero.

    Compiler owns physical binding and workspace capacity. Cirq owns allocation
    and release during export. Newly allocated output qubits remain live in the
    supplied manager and become the caller's responsibility. Input arrays use
    the register shape followed by its ordered bit axis.
    """
    cirq = _cirq()
    lefts = tuple(signature.lefts())
    rights = tuple(signature.rights())
    if input_quregs is not None and set(input_quregs) != {reg.name for reg in lefts}:
        raise ValueError("Cirq input placement must bind exactly the left registers")
    input_qubits: list[Qid] = []
    for register in lefts:
        shape = _register_shape(register)
        if input_quregs is None:
            qubits = _named_qubits(cirq, register, shape)
        else:
            supplied = input_quregs[register.name]
            qubits = np.asarray(
                [supplied] if isinstance(supplied, cirq.Qid) else supplied, dtype=object
            )
        if qubits.shape != shape:
            raise ValueError(f"Cirq register {register.name!r} requires shape {shape}")
        input_qubits.extend(qubits.flat)
    if any(not isinstance(q, cirq.Qid) or q.dimension != 2 for q in input_qubits):
        raise ValueError("Cirq inputs require two-dimensional qubits")
    if len(set(input_qubits)) != len(input_qubits):
        raise ValueError("Cirq inputs require distinct qubits")
    if len(input_qubits) > bound.circuit.n_wires or bound.classical_outputs:
        raise ValueError("Cirq export requires bound quantum register ports")
    output_shapes = tuple(_register_shape(register) for register in rights)
    expected_widths = tuple(
        register.bitsize for register in rights for _ in register.all_idxs()
    )
    if tuple(map(len, bound.outputs)) != expected_widths:
        raise ValueError("Cirq output widths must match the register signature")
    output_wires = tuple(wire for word in bound.outputs for wire in word)
    if len(set(output_wires)) != len(output_wires) or any(
        type(wire) is not int or not 0 <= wire < bound.circuit.n_wires
        for wire in output_wires
    ):
        raise ValueError("Cirq outputs require distinct in-range bound wires")
    offsets: list[int] = []
    exported = _export_cirq(
        bound.circuit,
        allow_lossy=False,
        extent_declared_by_gate=True,
        include_inverse_phases=True,
        feedback_key_prefix=feedback_key_prefix,
        instruction_offsets=offsets,
    )
    consumed = frozenset(
        wire
        for event in bound.events
        if event.kind == "free"
        for wire in event.wires
        if wire < len(input_qubits)
    )
    operations, returned = _replay_workspace_operations(
        tuple(exported.circuit.all_operations()),
        bound.events,
        input_qubits,
        consumed,
        (),
        qubit_manager,
        instruction_offsets=offsets,
        retained_outputs=output_wires,
    )
    outputs: dict[str, NDArray[np.object_]] = {}
    position = 0
    for register, shape in zip(rights, output_shapes, strict=True):
        size = int(np.prod(shape))
        outputs[register.name] = np.asarray(
            returned[position : position + size], dtype=object
        ).reshape(shape)
        position += size
    assert isinstance(exported.report, CirqConversionReport)
    return (
        CirqExportResult(
            cirq.FrozenCircuit(cirq.Moment([op]) for op in operations), exported.report
        ),
        outputs,
    )
