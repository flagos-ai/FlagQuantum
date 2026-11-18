"""Convert concrete external sparse oracles to owned native snapshots.

The optional dependency ends here. Independent logical models are supplied by
the caller; emitted primitive gates never manufacture a source cost model.
"""

from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

import numpy as np

from ...algorithms.primitives.circuit_sparse_oracles import (
    CircuitColumnOracle,
    CircuitSqrtEntryOracle,
)
from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.primitives.circuit_binding import restore_output_order_ir
from ...core.ir import CircuitIR
from ._feedback_import import import_cirq_feedback
from .conversion import _cirq, _lower_gate


class _CleanAllocations:
    """Track explicit clean allocation and reject borrowed or unknown storage."""

    def __init__(self, cirq: Any, inputs: tuple[Any, ...]) -> None:
        self.manager = cirq.GreedyQubitManager(
            prefix="oracle_work", maximize_reuse=True
        )
        self.inputs = set(inputs)
        self.allocated: list[Any] = []

    def qalloc(self, n: int, dim: int = 2) -> list[Any]:
        if dim != 2:
            raise ValueError("Imported oracle workspace must be binary")
        result = list(self.manager.qalloc(n, dim=dim))
        for qubit in result:
            if qubit not in self.allocated:
                self.allocated.append(qubit)
        return result

    def qfree(self, qubits: Iterable[Any]) -> None:
        qubits = tuple(qubits)
        if not set(qubits) <= self.inputs | set(self.allocated):
            raise ValueError("Oracle released undeclared quantum storage")
        self.manager.qfree([q for q in qubits if q not in self.inputs])

    def qborrow(self, n: int, dim: int = 2) -> list[Any]:
        raise ValueError(
            "Imported sparse oracles require clean work, not dirty borrowing"
        )


def _width(source: Any) -> int:
    value = source.system_bitsize
    if type(value) is not int or value < 1:
        raise ValueError(
            "Foreign oracle binding requires a positive concrete index width"
        )
    return value


def _snapshot(source: Any, ports: tuple[tuple[str, int], ...]) -> tuple[CircuitIR, int]:
    cirq = _cirq()
    for side in (source.signature.lefts(), source.signature.rights()):
        side = tuple(side)
        actual = tuple((reg.name, reg.bitsize) for reg in side)
        if actual != ports or any(reg.shape for reg in side):
            raise ValueError(
                "Foreign oracle must have the declared scalar input/output ports"
            )
    incoming = tuple(cirq.LineQubit.range(sum(width for _, width in ports)))
    arrays: dict[str, Any] = {}
    offset = 0
    for name, width in ports:
        arrays[name] = np.asarray(incoming[offset : offset + width], dtype=object)
        offset += width
    manager = _CleanAllocations(cirq, incoming)
    register_names = [reg.name for reg in source.signature]
    if len(register_names) != len(set(register_names)):
        # The frozen gate wrapper rejects same-name LEFT/RIGHT registers.
        # Its Composite exporter supports the original sided column signature.
        operation, returned = source.decompose_bloq().to_cirq_circuit_and_quregs(
            qubit_manager=manager, **arrays
        )
    else:
        operation, returned = source.as_cirq_op(qubit_manager=manager, **arrays)
    if set(returned) != {name for name, _ in ports}:
        raise ValueError("Foreign oracle returned different logical ports")
    outgoing: list[Any] = []
    for name, width in ports:
        array = np.asarray(returned[name], dtype=object)
        if array.shape != (width,):
            raise ValueError("Foreign oracle returned a different port shape")
        outgoing.extend(array.tolist())
    if len(set(outgoing)) != len(incoming):
        raise ValueError("Foreign oracle returned overlapping quantum outputs")

    def admitted(op: Any) -> bool:
        if op.tags:
            return True  # The strict importer reports uninterpreted tags.
        bare = op.without_classical_controls()
        if cirq.is_measurement(bare) or isinstance(
            bare.gate, (cirq.ResetChannel, cirq.GlobalPhaseGate)
        ):
            return True
        return _lower_gate(cirq, bare.gate, [], 0) is not None

    operations = (
        ()
        if operation is None
        else cirq.decompose(
            operation, keep=admitted, context=cirq.DecompositionContext(manager)
        )
    )
    primitive = cirq.Circuit(operations)
    order = (*incoming, *(q for q in manager.allocated if q not in incoming))
    if not (set(primitive.all_qubits()) | set(outgoing)) <= set(order):
        raise ValueError("Foreign oracle used undeclared external quantum inputs")
    imported = import_cirq_feedback(primitive, qubit_order=order).ir
    # Every output becomes the caller's original through port. Remaining owned
    # clean storage stays in the workspace partition, including returned ancillas.
    canonical = (*outgoing, *(q for q in order if q not in outgoing))
    swaps = restore_output_order_ir(tuple(order.index(q) for q in canonical))
    return replace(
        imported, instructions=(*imported.instructions, *swaps)
    ).validate(), len(swaps)


def import_column_oracle(
    source: object, *, models: tuple[ComponentCost, ComponentCost]
) -> CircuitColumnOracle:
    """Import a concrete column object's actual forward and adjoint operations."""
    foreign = cast(Any, source)
    width = _width(foreign)
    ports = (("l", width), ("i", width))
    forward, first = _snapshot(foreign, ports)
    backward, second = _snapshot(foreign.adjoint(), ports)
    return CircuitColumnOracle(
        width, foreign.num_nonzero, forward, backward, models, (first, second)
    )


def import_sqrt_entry_oracle(
    source: object, *, models: tuple[ComponentCost, ComponentCost]
) -> CircuitSqrtEntryOracle:
    """Import a concrete entry, retaining independent models and precision."""
    foreign = cast(Any, source)
    width = _width(foreign)
    ports = (("q", 1), ("i", width), ("j", width))
    forward, first = _snapshot(foreign, ports)
    backward, second = _snapshot(foreign.adjoint(), ports)
    return CircuitSqrtEntryOracle(
        width, forward, backward, models, foreign.epsilon, (first, second)
    )
