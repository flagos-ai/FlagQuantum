"""Check typed GF2 graph, ordered bindings and executable serialization."""

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as numpy
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import json
from importlib.metadata import version

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.physical_codec import native_program_to_dict
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.errors import SerializationError
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from tests.unit.register_source_graph import source_line_qubit_index
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 3, 4])
@pytest.mark.parametrize("kind", ["register", "zero", "nonzero"])
def test_typed_field_addition_matches_order_and_executes_roundtrip(
    width: int, kind: str
) -> None:
    assert version("qualtran") == "0.7.0"
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    constant = (
        None if kind == "register" else 0 if kind == "zero" else (1 << (width - 1)) | 1
    )
    source = GF2Addition(width) if constant is None else GF2AddK(width, constant)
    descriptor = binary_field_encoding(width).register_type()
    record = field_addition_program(descriptor, constant=constant)
    names = ("x", "y") if constant is None else ("x",)
    functions = {fn.name: fn for fn in record.functions}
    signature = functions["main"].registers
    assert signature is not None
    assert tuple(r.name for r in signature) == names
    assert all(r.dtype is descriptor for r in signature)
    assert [(r.name, r.bitsize) for r in signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    direct = []
    for op in functions["main"].blocks[0].operations:
        assert op.callee is not None
        leaf = functions[op.callee]
        assert leaf.registers is not None
        direct.append(
            {"field_x": "XGate", "field_cx": "CNOT"}.get(
                leaf.name, "Join" if leaf.registers[0].shape else "Split"
            )
        )
    original = source.decompose_bloq()
    assert direct == [
        type(node.bloq).__name__
        for node in sorted(original.bloq_instances, key=lambda node: node.i)
    ]

    x = (
        tuple(reversed(range(0, 2 * width, 2)))
        if constant is None
        else tuple(reversed(range(width)))
    )
    y = tuple(range(1, 2 * width, 2))
    ports = (x, y) if constant is None else (x,)
    bound = bind_register_program(record, ports)
    cirq_ops, _ = original.to_cirq_circuit_and_quregs(
        qubit_manager=None,
        **{
            name: np.array([cirq.LineQubit(i) for i in wires])
            for name, wires in zip(names, ports, strict=True)
        },
    )
    gate_names: dict[cirq.Gate | None, str] = {cirq.X: "x", cirq.CNOT: "cx"}
    assert_source(
        bound.circuit.instructions,
        tuple(
            Instruction(
                gate_names[op.gate],
                tuple(source_line_qubit_index(qubit) for qubit in op.qubits),
            )
            for op in cirq_ops.all_operations()
        ),
    )
    assert [gate.name for gate in bound.circuit.instructions] == (
        ["cx"] * width if constant is None else ["x"] * constant.bit_count()
    )
    assert bound.outputs == ports and not bound.events
    assert get_cost_value(source, QECGatesCost()).clifford == len(
        bound.circuit.instructions
    )
    ir = CircuitIR.from_dict(json.loads(json.dumps(bound.circuit.to_dict())))
    assert ir == bound.circuit
    with pytest.raises(SerializationError, match="requires a physical context"):
        native_program_to_dict(record)

    wire_count = bound.circuit.n_wires
    axis_map = tuple(i for word in ports for i in word)
    axes = tuple(axis_map.index(i) for i in range(wire_count))
    values = torch.arange(1, (1 << wire_count) + 1, dtype=torch.float64)
    ket = torch.complex(values, values.flip(0) - 3)
    ket /= ket.norm()
    input_logical = ket.reshape((2,) * wire_count).permute(axis_map).flatten()
    original_tensor = torch.as_tensor(source.tensor_contract().copy(), dtype=ket.dtype)
    expected = (
        (original_tensor @ input_logical)
        .reshape((2,) * wire_count)
        .permute(axes)
        .flatten()
    )
    circuit = Circuit(wire_count, inputs=ket, dtype=ket.dtype).append(
        FieldOperation(wire_count, ir.instructions)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-14, rtol=0)


def test_typed_addition_rejects_nonfield_and_invalid_constant() -> None:
    with pytest.raises(ValueError, match="binary field"):
        field_addition_program(unsigned_register_type(3))
    with pytest.raises(ValueError, match="must fit"):
        field_addition_program(binary_field_encoding(3).register_type(), constant=8)
