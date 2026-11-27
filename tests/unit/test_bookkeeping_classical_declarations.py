"""Classical storage widths survive source bookkeeping declarations and views."""

from __future__ import annotations

from math import prod
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
    auto_partition_signature,
)
from flagquantum.compiler.registers.layout.component_partition import (
    partition_component_program,
    partition_signature,
)
from flagquantum.compiler.registers.layout.partition import (
    partition_register,
    unpartition_register,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_program_classical_registers import identity

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_signatures(native: tuple[Register, ...], source: Any) -> None:
    """Compare actual metadata directly; the source CUInt wire codec rejects it."""
    try:
        from qualtran._infra.data_types import CUInt, QAny, QBit
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    assert len(native) == len(source)
    for actual, original in zip(native, source, strict=True):
        assert (actual.name, actual.shape_symbolic, actual.side.name) == (
            original.name,
            original.shape_symbolic,
            original.side.name,
        )
        assert (actual.dtype.num_qubits, actual.dtype.num_cbits) == (
            original.dtype.num_qubits,
            original.dtype.num_cbits,
        )
        if actual.dtype.parameters[:1] == (("storage", "classical"),):
            assert type(original.dtype) is CUInt
        elif actual.dtype.encoding == "opaque":
            assert type(original.dtype) is QAny
        else:
            assert actual.dtype == quantum_bit_type() and type(original.dtype) is QBit


@pytest.mark.parametrize("shape", ((), (2,), (0, 2)))
@pytest.mark.parametrize("reverse", (False, True))
def test_partition_total_storage_width_and_raw_sdk_order(
    shape: tuple[int, ...], reverse: bool
) -> None:
    try:
        from qualtran._infra.data_types import CUInt, QBit
        from qualtran._infra.registers import Register as OriginalRegister
        from qualtran.bloqs.bookkeeping.partition import Partition
        from qualtran.resource_counting._bloq_counts import QECGatesCost
        from qualtran.resource_counting._costing import get_cost_value
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    parts = (
        Register("c", classical_unsigned_register_type(2), shape),
        Register("tail", quantum_bit_type()),
    )
    width = 2 * prod(shape) + 1
    source = Partition(
        width,
        (OriginalRegister("c", CUInt(2), shape), OriginalRegister("tail", QBit())),
        partition=not reverse,
    )
    signature = partition_signature(parts, reverse=reverse)
    assert_signatures(tuple(signature), source.signature)
    assert signature[0].bitsize == width
    declaration = partition_component_program(parts, "view", reverse)
    assert verify_program_record(declaration) == ()
    if prod(shape):
        assert declaration.functions[0].blocks == ()
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(declaration, (), borrowed=(0,), entry="view")
    else:
        bound = bind_register_program(declaration, ((0,),), entry="view")
        assert (
            bound.outputs == ((0,),)
            and not bound.events
            and not bound.circuit.instructions
        )
    layout = (("c", width - 1), ("tail", 1))
    wires = tuple(reversed(range(width)))
    split = partition_register(wires, layout)
    if reverse:
        operation, actual = source.as_cirq_op(
            None,
            c=np.asarray(split["c"]).reshape((*shape, 2)),
            tail=np.asarray(split["tail"]),
        )
        assert tuple(actual["x"]) == unpartition_register(split, layout) == wires
    else:
        operation, actual = source.as_cirq_op(None, x=np.asarray(wires))
        assert tuple(actual["c"].flat) == split["c"]
        assert tuple(actual["tail"]) == split["tail"]
    assert operation is None
    assert get_cost_value(source, QECGatesCost()).asdict() == {}


@pytest.mark.parametrize("shape", ((), (2,), (0, 2)))
@pytest.mark.parametrize("left_only", (False, True))
def test_auto_partition_retains_actual_classical_identity_call_graph(
    shape: tuple[int, ...], left_only: bool
) -> None:
    try:
        import cirq
        from qualtran._infra.composite_bloq import BloqBuilder
        from qualtran._infra.data_types import CUInt, QAny
        from qualtran._infra.registers import Register as OriginalRegister
        from qualtran._infra.registers import Signature as OriginalSignature
        from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original
        from qualtran.bloqs.bookkeeping.auto_partition import Unused
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    parts = (Register("c", classical_unsigned_register_type(2), shape),)
    signature = Signature(parts)
    source_builder, inputs = BloqBuilder.from_signature(
        OriginalSignature((OriginalRegister("c", CUInt(2), shape),))
    )
    child = source_builder.finalize(**inputs)
    width = 2 * prod(shape)
    padding = 0 if left_only else 1
    segments = (("c", width),) if left_only else (("c", width), (None, padding))
    layout = AutoPartition((("word", segments),), left_only=left_only)
    source = Original(
        child,
        (
            (
                OriginalRegister("word", QAny(width + padding)),
                ("c",) if left_only else ("c", Unused(padding)),
            ),
        ),
        left_only=left_only,
    )
    native = auto_partition_signature(signature, layout)
    assert_signatures(tuple(native), source.signature)
    module = auto_partition_component_program(
        ProgramRecord((identity("main", parts),)), layout
    )
    assert verify_program_record(module) == ()
    root = next(f for f in module.functions if f.name == "packed")
    definitions = {f.name: f for f in module.functions}
    original = source.decompose_bloq()
    calls = root.blocks[0].operations
    assert len(calls) == len(original.bloq_instances) == (2 if left_only else 3)
    # Creation order is stable even when an empty child has no DAG edges.
    instances = sorted(original.bloq_instances, key=lambda instance: instance.i)
    for call, instance in zip(calls, instances, strict=True):
        registers = definitions[call.callee or ""].registers
        assert registers is not None
        assert_signatures(registers, instance.bloq.signature)
    count = prod(shape)
    assert calls[1].operands == calls[0].results[:count]
    if not left_only:
        assert calls[2].operands == (*calls[1].results, *calls[0].results[count:])
    assert len(calls[1].operands) == len(calls[1].results) == count
    if prod(shape):
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(
                module,
                (tuple(range(width + padding)),),
                entry="packed",
                borrowed=(width + padding,),
            )
    else:
        physical = bind_register_program(
            module,
            (tuple(range(width + padding)),),
            entry="packed",
            borrowed=(width + padding,),
        )
        assert physical.outputs == (
            () if left_only else (tuple(range(width + padding)),)
        )
        assert not physical.circuit.instructions and not physical.events
    if width:
        with pytest.raises(ValueError, match="classical wires"):
            original.to_cirq_circuit(
                cirq_quregs={"word": np.asarray(cirq.LineQubit.range(width + padding))},
                qubit_manager=cirq.SimpleQubitManager(),
            )
    # This is the existing raw layout protocol, not execution of classical slots.
    wires = tuple(range(width + padding))
    bound, unused = layout.bind({"word": wires})
    restored = layout.restore(bound, unused)
    assert restored == ({"c": wires} if left_only else {"word": wires})
    if width + padding:
        from examples.algorithms.binary_field_arithmetic import FieldOperation
        from flagquantum import Circuit
        from flagquantum.simulation.unitary import get_unitary

        actual = get_unitary(
            Circuit(width + padding, dtype=torch.complex128).append(
                FieldOperation(width + padding, ())
            )
        )
        if prod(shape):
            expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
        else:
            # Original empty composite flattening drops the named array output.
            with pytest.raises(KeyError, match="c"):
                source.tensor_contract()
            torch.testing.assert_close(
                actual,
                torch.eye(1 << (width + padding), dtype=torch.complex128),
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("fraction", (-1, -2, -7))
@pytest.mark.parametrize("shape", ((), (2,), (0, 2)))
@pytest.mark.parametrize("left_only", (False, True))
def test_auto_partition_signed_empty_storage_full_phase_and_owned_outputs(
    fraction: int, shape: tuple[int, ...], left_only: bool
) -> None:
    try:
        from qualtran._infra.composite_bloq import BloqBuilder
        from qualtran._infra.data_types import QAny, QBit, QFxp
        from qualtran._infra.registers import Register as OriginalRegister
        from qualtran._infra.registers import Signature as OriginalSignature
        from qualtran.bloqs.basic_gates.hadamard import Hadamard
        from qualtran.bloqs.basic_gates.t_gate import TGate
        from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original
        from qualtran.resource_counting._bloq_counts import QECGatesCost
        from qualtran.resource_counting._costing import get_cost_value
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program.register_types import fixed_point_register_type
    from flagquantum.simulation.unitary import get_unitary
    from tests.unit.test_subtraction_foreign_file import native_signature

    source_dtype = QFxp(0, fraction, signed=True)
    builder, inputs = BloqBuilder.from_signature(
        OriginalSignature(
            (OriginalRegister("q", source_dtype, shape), OriginalRegister("s", QBit()))
        )
    )
    s = builder.add(Hadamard(), q=inputs["s"])
    s = builder.add(TGate(), q=s)
    child = builder.finalize(q=inputs["q"], s=s)
    source = Original(
        child,
        (
            (OriginalRegister("a", QAny(0)), ("q",)),
            (OriginalRegister("b", QAny(1)), ("s",)),
        ),
        left_only=left_only,
    )
    dtype = fixed_point_register_type(0, fraction, True)
    signature = Signature(
        (Register("q", dtype, shape), Register("s", quantum_bit_type()))
    )
    leaves = tuple(
        bit_gate_function(
            op,
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction(op, (0,)),)),
        )
        for op in ("h", "t")
    )
    native = ComponentProgramBuilder(signature, leaves)
    s = native.call_dict("h", {"q": native.inputs["s"]})["q"]
    s = native.call_dict("t", {"q": s})["q"]
    record = native.finalize({"q": native.inputs["q"], "s": s})
    module = auto_partition_component_program(
        record,
        AutoPartition((("a", (("q", 0),)), ("b", (("s", 1),))), left_only=left_only),
    )
    packed = next(f for f in module.functions if f.name == "packed")
    assert packed.registers == tuple(native_signature(source.signature))
    bound = bind_register_program(module, ((), (0,)), entry="packed")
    assert bound.outputs == (((),) * prod(shape) + ((0,),) if left_only else ((), (0,)))
    assert bound.events == ()
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == [
        ("h", (0,)),
        ("t", (0,)),
    ]
    actual = get_unitary(
        Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, bound.circuit.instructions)
        )
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == model.t == 1 and model.measurement == 0
