"""Capture an external component with concrete ports into owned primitive IR."""

from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

import numpy as np

from ...compiler.registers.primitives.circuit_binding import restore_output_order_ir
from ...core.ir import CircuitIR
from ._feedback_import import (
    _has_explicit_matrix,
    _lower_feedback_gate,
    import_cirq_feedback,
)
from ._integer_arithmetic import _square_operations
from ._modular_arithmetic import _mod_add_operations, _private_modular_keys
from ._multiplication_arithmetic import (
    multiplication_operations,
    multiplication_program,
)
from ._phase_arithmetic import _phase_add_operations
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


def _snapshot(
    source: Any,
    ports: tuple[tuple[str, int], ...],
    *,
    local_ports: tuple[str, ...] = (),
    scalar_carrier: bool = False,
    allow_empty_ports: bool = False,
) -> tuple[CircuitIR, int]:
    from qualtran._infra.adjoint import Adjoint
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.controlled import Controlled, CtrlSpec
    from qualtran.bloqs.arithmetic.multiplication import (
        InvertRealNumber,
        MultiplyTwoReals,
        PlusEqualProduct,
        Product,
        ScaleIntByReal,
        Square,
        SquareRealNumber,
        SumOfSquares,
    )
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap
    from qualtran.bloqs.basic_gates.t_gate import TGate
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAddK
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
    from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate

    multiplication_types = (
        Product,
        PlusEqualProduct,
        SumOfSquares,
        ScaleIntByReal,
        MultiplyTwoReals,
        SquareRealNumber,
        InvertRealNumber,
    )
    cirq = _cirq()
    if type(scalar_carrier) is not bool or (scalar_carrier and ports):
        raise ValueError("Scalar carrier capture requires empty logical ports")
    if type(allow_empty_ports) is not bool:
        raise ValueError("Empty-port admission must be Boolean")
    local = set(local_ports)
    if len(local) != len(local_ports) or not local <= {name for name, _ in ports}:
        raise ValueError("Local ports must be distinct declared physical ports")
    local_registers = tuple(reg for reg in source.signature if reg.name in local)
    if len(local_registers) != len(local) or any(
        reg.side.value not in (1, 2) for reg in local_registers
    ):
        raise ValueError("Local storage requires one-sided source registers")
    shapes: dict[str, tuple[int, ...]] = {}
    for side in (source.signature.lefts(), source.signature.rights()):
        side = tuple(side)
        if any(
            type(reg.bitsize) is not int
            or reg.bitsize < (0 if allow_empty_ports and reg.side.value == 3 else 1)
            or any(type(size) is not int or size <= 0 for size in reg.shape)
            for reg in side
        ):
            raise ValueError("Foreign component ports need concrete positive shapes")
        actual = tuple(
            (reg.name, reg.total_bits()) for reg in side if reg.name not in local
        )
        layout = {reg.name: (*reg.shape, reg.bitsize) for reg in side}
        through = tuple((name, width) for name, width in ports if name not in local)
        if actual != through or any(
            shapes[name] != shape for name, shape in layout.items() if name in shapes
        ):
            raise ValueError(
                "Foreign component must have matching declared input/output ports"
            )
        shapes.update(layout)
    if tuple((reg.name, reg.total_bits()) for reg in local_registers) != tuple(
        (name, width) for name, width in ports if name in local
    ):
        raise ValueError("Local port widths must match the source declaration")
    incoming = tuple(cirq.LineQubit.range(sum(width for _, width in ports)))
    arrays: dict[str, Any] = {}
    offset = 0
    for name, width in ports:
        arrays[name] = np.asarray(
            incoming[offset : offset + width], dtype=object
        ).reshape(shapes[name])
        offset += width
    manager = _CleanAllocations(cirq, incoming)
    input_arrays = {reg.name: arrays[reg.name] for reg in source.signature.lefts()}
    register_names = [reg.name for reg in source.signature]
    if len(register_names) != len(set(register_names)):
        # The frozen gate wrapper rejects same-name LEFT/RIGHT registers.
        # Its Composite exporter supports the original sided column signature.
        operation, returned = source.decompose_bloq().to_cirq_circuit_and_quregs(
            qubit_manager=manager, **input_arrays
        )
    else:
        operation, returned = source.as_cirq_op(qubit_manager=manager, **input_arrays)
    if set(returned) != {reg.name for reg in source.signature.rights()}:
        raise ValueError("Foreign oracle returned different logical ports")
    returned_qubits = {
        q for value in returned.values() for q in np.asarray(value, dtype=object).flat
    }
    consumed_storage = iter(q for q in incoming if q not in returned_qubits)
    outgoing: list[Any] = []
    for name, width in ports:
        # A consumed local input can become a different logical output through
        # source bookkeeping aliases. Use the freed input storage, not that
        # possibly still-live input identity, as the physical clean output.
        if name in returned:
            value = returned[name]
        else:
            value = np.asarray(
                [next(consumed_storage) for _ in range(width)], dtype=object
            ).reshape(shapes[name])
        array = np.asarray(value, dtype=object)
        if array.shape != shapes[name]:
            raise ValueError("Foreign oracle returned a different port shape")
        outgoing.extend(array.reshape(-1).tolist())
    if len(set(outgoing)) != len(incoming):
        raise ValueError("Foreign oracle returned overlapping quantum outputs")

    def admitted(op: Any) -> bool:
        if op.tags:
            return True  # The strict importer reports uninterpreted tags.
        bare = op.without_classical_controls()
        arithmetic = (
            bare.gate.bloq if isinstance(bare.gate, BloqAsCirqGate) else bare.gate
        )
        root = arithmetic.subbloq if isinstance(arithmetic, Adjoint) else arithmetic
        if type(root) in multiplication_types:
            return False
        if cirq.is_measurement(bare) or isinstance(
            bare.gate, (cirq.ResetChannel, cirq.GlobalPhaseGate)
        ):
            return True
        return (
            _has_explicit_matrix(cirq, bare.gate)
            or _lower_feedback_gate(cirq, bare.gate, [], 0) is not None
        )

    decomposition_manager = InteropQubitManager(manager)
    decomposition_manager.manage_qubits(incoming)
    # The optional SDK annotates its nominal base class. This validated adapter
    # implements the allocation protocol and reserves live source-frame Qids.
    sdk_manager: Any = decomposition_manager
    modular_calls = 0
    phase_calls = 0
    square_calls = 0
    multiplication_calls = 0
    modular_tag = object()

    def leaf_operation(op: Any) -> Any:
        """Use a source leaf's concrete operation only with unchanged ports."""
        nonlocal modular_calls, phase_calls, square_calls, multiplication_calls
        if op.tags:
            return NotImplemented
        arithmetic = op.gate.bloq if isinstance(op.gate, BloqAsCirqGate) else op.gate
        square = arithmetic.subbloq if isinstance(arithmetic, Adjoint) else arithmetic
        if type(square) in multiplication_types:
            multiplication_calls += 1
            record = multiplication_program(
                square, adjoint=isinstance(arithmetic, Adjoint)
            )
            return multiplication_operations(
                record,
                tuple(op.qubits),
                decomposition_manager,
                f"multiplication_{multiplication_calls}",
                modular_tag,
            )
        if type(square) is Square:
            square_calls += 1
            return _square_operations(
                square,
                tuple(op.qubits),
                decomposition_manager,
                f"square_{square_calls}",
                modular_tag,
            )
        if (
            isinstance(arithmetic, Adjoint)
            and type(arithmetic.subbloq) is AddIntoPhaseGrad
        ):
            arithmetic = arithmetic.subbloq.adjoint()
        if type(arithmetic) is AddIntoPhaseGrad:
            phase_calls += 1
            return _phase_add_operations(
                arithmetic,
                tuple(op.qubits),
                decomposition_manager,
                f"phase_add_{phase_calls}",
                modular_tag,
            )
        if isinstance(arithmetic, Adjoint) and type(arithmetic.subbloq) is ModAddK:
            arithmetic = arithmetic.subbloq**-1
        if type(arithmetic) is ModAddK:
            modular_calls += 1
            return _mod_add_operations(
                arithmetic,
                tuple(op.qubits),
                decomposition_manager,
                f"mod_add_{modular_calls}",
                modular_tag,
            )
        if (
            type(arithmetic) is Controlled
            and type(arithmetic.subbloq) is TGate
            and arithmetic.ctrl_spec == CtrlSpec()
        ):
            # The source declares this controlled phase as an atomic leaf.
            phase, _ = arithmetic.subbloq.as_cirq_op(
                qubit_manager=sdk_manager, q=op.qubits[1:]
            )
            return phase.controlled_by(op.qubits[0])
        leaf = op.gate.bloq if isinstance(op.gate, BloqAsCirqGate) else None
        if leaf is None:
            return NotImplemented
        if type(leaf) is TwoBitCSwap:
            # The atomic source exposes its exact recipe outside Cirq's
            # decomposition protocol. Use that recipe with its named operands.
            recipe = leaf.to_clifford_t_circuit()
            return cirq.Circuit(recipe).transform_qubits(
                dict(
                    zip(
                        map(cirq.NamedQubit, ("ctrl", "x", "y")), op.qubits, strict=True
                    )
                )
            )
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
        replacement, outputs = leaf.as_cirq_op(qubit_manager=sdk_manager, **quregs)
        if replacement is None or replacement == op:
            return NotImplemented
        if set(outputs) != set(quregs) or any(
            not np.array_equal(outputs[name], value) for name, value in quregs.items()
        ):
            return NotImplemented
        return replacement

    def expand(tree: Any) -> Iterator[Any]:
        # Export constructs sibling operations before recursively decomposing
        # them. Its allocator may already have freed a RIGHT output that is
        # still live between the producing and consuming sibling operations.
        frame = tuple(cirq.flatten_to_ops(tree))
        # All sibling Qids belong to this already exported circuit frame.
        # Later siblings can own a prepared flag even when the exporter has
        # eagerly freed it; descendants must not allocate that live Qid.
        with decomposition_manager.reserve_qubits(q for op in frame for q in op.qubits):
            live_outputs: set[Any] = set()
            for op in frame:
                if admitted(op):
                    yield op
                    continue
                leaf = op.gate.bloq if isinstance(op.gate, BloqAsCirqGate) else op.gate
                consumed: set[Any] = set()
                produced: set[Any] = set()
                if isinstance(leaf, Bloq):
                    offset = 0
                    for reg in leaf.signature:
                        count = int(reg.total_bits())
                        qubits = op.qubits[offset : offset + count]
                        if reg.side.value == 1:
                            consumed.update(qubits)
                        elif reg.side.value == 2:
                            produced.update(qubits)
                        offset += count
                    if offset != len(op.qubits):
                        raise ValueError(
                            "Foreign operation does not match its register extent"
                        )
                with decomposition_manager.reserve_qubits((*op.qubits, *live_outputs)):
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
                live_outputs.difference_update(consumed)
                live_outputs.update(produced)

    operations = () if operation is None else expand(operation)
    # Preserve the source's expanded operation order. Conversion must not add
    # a second scheduling pass that interleaves atomic compute/erase recipes.
    primitive = cirq.Circuit(operations, strategy=cirq.InsertStrategy.NEW)
    if modular_calls or phase_calls or square_calls or multiplication_calls:
        primitive = _private_modular_keys(cirq, primitive, modular_tag)
    # A fresh RIGHT output can use the caller's unused clean local slot by
    # renaming its entire lifetime. This is bookkeeping, not a physical SWAP.
    # If the source aliases an original input instead, preserve that alias and
    # let the existing explicit output-order restoration bind it below.
    renames: dict[Any, Any] = {}
    used = set(primitive.all_qubits())
    for reg in local_registers:
        if reg.side.value == 2:
            for actual, slot in zip(
                np.asarray(returned[reg.name], dtype=object).flat,
                arrays[reg.name].flat,
                strict=True,
            ):
                if actual not in incoming and slot not in used:
                    renames[actual] = slot
    if renames:
        primitive = primitive.transform_qubits(lambda q: renames.get(q, q))
        outgoing = [renames.get(q, q) for q in outgoing]
    live = set(primitive.all_qubits()) | set(outgoing)
    order = (
        *incoming,
        *(
            q
            for q in manager.allocated
            if q not in incoming and q not in renames
            # The SDK's RIGHT-output relocation can leave allocation names
            # that occur in no operation or output after that relocation.
            and (not local or q in live)
        ),
    )
    if not order and scalar_carrier:
        # Explicitly requested storage represents a wire-free source scalar.
        # The existing IR requires one wire even for an empty identity body.
        # Phase expansion preserves any state on this carrier.
        order = tuple(manager.qalloc(1))
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
