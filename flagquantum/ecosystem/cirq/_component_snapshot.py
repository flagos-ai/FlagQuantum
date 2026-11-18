"""Capture an external scalar-port component into owned primitive IR."""

from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

import numpy as np

from ...compiler.registers.primitives.circuit_binding import restore_output_order_ir
from ...core.ir import CircuitIR
from ._feedback_import import _lower_feedback_gate, import_cirq_feedback
from ._qubit_manager import InteropQubitManager
from .conversion import _cirq


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
        return _lower_feedback_gate(cirq, bare.gate, [], 0) is not None

    decomposition_manager = InteropQubitManager(manager)
    decomposition_manager.manage_qubits(incoming)

    def leaf_operation(op: Any) -> Any:
        """Use a source leaf's concrete operation only with unchanged ports."""
        if op.tags:
            return NotImplemented
        leaf = getattr(op.gate, "bloq", None)
        if leaf is None:
            return NotImplemented
        registers = tuple(leaf.signature)
        if any(reg.side.value != 3 for reg in registers):
            return NotImplemented
        quregs: dict[str, Any] = {}
        offset = 0
        for reg in registers:
            count = int(reg.total_bits())
            quregs[reg.name] = np.asarray(
                op.qubits[offset : offset + count], dtype=object
            ).reshape((*reg.shape, int(reg.bitsize)))
            offset += count
        replacement, outputs = leaf.as_cirq_op(
            qubit_manager=decomposition_manager, **quregs
        )
        if replacement is None or replacement == op:
            return NotImplemented
        if set(outputs) != set(quregs) or any(
            not np.array_equal(outputs[name], value) for name, value in quregs.items()
        ):
            return NotImplemented
        return replacement

    def expand(tree: Any) -> Iterator[Any]:
        for op in cirq.flatten_to_ops(tree):
            if admitted(op):
                yield op
                continue
            with decomposition_manager.reserve_qubits(op.qubits):
                child = leaf_operation(op)
                if child is NotImplemented:
                    child = cirq.decompose_once(
                        op,
                        context=cirq.DecompositionContext(decomposition_manager),
                        default=None,
                    )
                if child is None:
                    raise ValueError(
                        f"Foreign oracle primitive cannot be decomposed: {op!r}"
                    )
                yield from expand(child)

    operations = () if operation is None else expand(operation)
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
