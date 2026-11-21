# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source graph expansion follows Qualtran v0.7.0 CompositeBloq Cirq export.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_bloq_to_cirq.py

"""Expand actual through-register source graphs and restore physical aliases."""

from typing import Any

import numpy as np

from ...compiler.registers.primitives.circuit_binding import restore_output_order_ir
from .conversion import _cirq


def source_graph_operations(operation: Any, manager: Any) -> tuple[Any, ...] | None:
    """Prefer the source's typed graph over its independent Cirq gate wrapper."""
    source = getattr(operation.gate, "bloq", operation.gate)
    if not callable(getattr(source, "decompose_bloq", None)):
        return None
    from qualtran import (
        Bloq,
        CompositeBloq,
        DecomposeNotImplementedError,
        DecomposeTypeError,
    )
    from qualtran._infra.gate_with_registers import split_qubits

    if not isinstance(source, Bloq) or any(
        reg.side.value != 3 for reg in source.signature
    ):
        return None
    try:
        graph = (
            source if isinstance(source, CompositeBloq) else source.decompose_bloq()
        ).flatten()
    except (DecomposeNotImplementedError, DecomposeTypeError):
        return None
    inputs = split_qubits(source.signature, operation.qubits)
    circuit, outputs = graph.to_cirq_circuit_and_quregs(qubit_manager=manager, **inputs)
    if set(inputs) != set(outputs):
        raise ValueError("Source through graph returned different register names")
    outgoing = tuple(
        qubit
        for reg in source.signature
        for qubit in np.asarray(outputs[reg.name], dtype=object).flat
    )
    if len(outgoing) != len(operation.qubits) or len(set(outgoing)) != len(outgoing):
        raise ValueError("Source through graph returned invalid output aliases")
    if outgoing == tuple(operation.qubits):
        return tuple(circuit.all_operations())
    # Export may return a word on rotated inputs or fresh storage. Include every
    # temporary in the physical permutation so its zero/dirty state is retained.
    order = (
        *operation.qubits,
        *sorted(
            (circuit.all_qubits() | set(outgoing)) - set(operation.qubits), key=str
        ),
    )
    canonical = (*outgoing, *(q for q in order if q not in outgoing))
    swaps = restore_output_order_ir(tuple(order.index(q) for q in canonical))
    cirq = _cirq()
    return (
        *circuit.all_operations(),
        *(cirq.SWAP(*(order[wire] for wire in gate.wires)) for gate in swaps),
    )
