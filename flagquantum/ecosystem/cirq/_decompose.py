# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_interop/decompose_protocol.py.

"""Evaluate a single Cirq decomposition within the optional SDK boundary."""

from typing import Any
from uuid import uuid4

from ._qubit_manager import InteropQubitManager
from .conversion import _cirq, _import_static_cirq
from .models import CirqImportResult


def decompose_cirq_once(value: Any) -> list[Any] | None:
    """Return the leaves in protocol order, or None when unavailable."""
    cirq = _cirq()
    manager = InteropQubitManager(
        cirq.GreedyQubitManager(prefix=f"_{uuid4()}", maximize_reuse=True)
    )
    if isinstance(value, cirq.Operation):
        manager.manage_qubits(value.qubits)
    context = cirq.DecompositionContext(qubit_manager=manager)
    if isinstance(value, cirq.Gate):
        qubits = cirq.LineQid.for_gate(value)
        manager.manage_qubits(qubits)
        tree = cirq.decompose_once_with_qubits(
            value,
            qubits,
            context=context,
            flatten=False,
            default=None,
        )
    else:
        tree = cirq.decompose_once(value, context=context, flatten=False, default=None)
    return None if tree is None else list(cirq.flatten_to_ops(tree))


def import_cirq_decomposition(value: Any) -> CirqImportResult | None:
    """Import one explicitly requested decomposition in input operand order.

    An empty op tree still owns its complete logical wire extent. Extra wires
    require a lifetime mapping before they can enter native IR.
    """
    cirq = _cirq()
    if isinstance(value, cirq.Gate):
        value = value.on(*cirq.LineQid.for_gate(value))
    if not isinstance(value, cirq.GateOperation) or value.tags:
        raise TypeError("Cirq decomposition import requires an untagged gate operation")
    wire_order = tuple(value.qubits)
    if (
        not wire_order
        or len(set(wire_order)) != len(wire_order)
        or any(qubit.dimension != 2 for qubit in wire_order)
    ):
        raise ValueError(
            "Cirq decomposition import requires distinct two-dimensional wires"
        )
    operations = decompose_cirq_once(value)
    if operations is None:
        return None
    bound = set(wire_order)
    if any(qubit not in bound for op in operations for qubit in op.qubits):
        raise ValueError(
            "Cirq decomposition allocates wires without a native lifetime binding"
        )
    circuit = cirq.Circuit(cirq.Moment([operation]) for operation in operations)
    return _import_static_cirq(cirq, circuit, wire_order=wire_order)
