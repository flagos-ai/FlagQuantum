"""Check typed AutoPartition declarations and real packed programs."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
    auto_partition_signature,
)
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source():
    from qualtran.bloqs.bookkeeping import auto_partition

    raw = Path(auto_partition.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "93cddd9cbb1ab309f713316f066f16d93e5fd8b0"
    )


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize(
    "left_only,padding,side",
    [
        (False, 0, RegisterSide.THRU),
        (False, 1, RegisterSide.THRU),
        (False, 0, RegisterSide.LEFT),
        (True, 0, RegisterSide.RIGHT),
    ],
)
def test_typed_outer_signature_matches_source(symbolic, left_only, padding, side):
    from qualtran import QUInt, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original
    from qualtran.bloqs.bookkeeping.auto_partition import Unused
    from sympy import Symbol

    width = Symbol("n", positive=True, integer=True) if symbolic else 3
    dtype = unsigned_register_type(width)
    signature = Signature((Register("x", dtype),))
    outer_dtype = unsigned_register_type(width + padding)
    outer = Register("packed", outer_dtype, side=side)
    segments = (("x", width), (None, padding)) if padding else (("x", width),)
    layout = AutoPartition((("packed", segments),), left_only=left_only)
    external = SourceRegister("packed", QUInt(width + padding), side=Side[side.name])
    parts = ["x", Unused(padding)] if padding else ["x"]
    source = Original(
        BitwiseNot(QUInt(width)), [(external, parts)], left_only=left_only
    )

    native = auto_partition_signature(signature, layout, packed_ports=(outer,))
    assert tuple(
        (r.name, r.shape_symbolic, r.bitsize, r.side.name) for r in native
    ) == tuple(
        (r.name, r.shape_symbolic, r.bitsize, r.side.name) for r in source.signature
    )
    assert native[0].dtype == outer_dtype
    if left_only:
        assert native[1].dtype == dtype


@pytest.mark.parametrize("left_only", [False, True])
def test_typed_packed_entry_binds_and_appends_original_bitwise_not(left_only):
    import cirq
    import numpy as np
    import torch
    from qualtran import QUInt, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition as Original
    from qualtran.bloqs.bookkeeping.auto_partition import Unused
    from qualtran.resource_counting import QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program.register_bindings import RegisterBinding
    from flagquantum.core.ir.program.register_types import quantum_bit_type

    width = 3
    outer_width = width if left_only else width + 1
    dtype = unsigned_register_type(width)
    x = bit_gate_function(
        "flip",
        (Register("q", quantum_bit_type()),),
        CircuitIR(n_wires=1, instructions=(Instruction("x", (0,)),)),
    )
    child = ComponentProgramBuilder(Signature((Register("x", dtype),)), (x,))
    word = child.inputs["x"]
    assert isinstance(word, RegisterBinding)
    bits = child.split(word)
    for i, bit in enumerate(bits):
        bits[i] = child.call("flip", {"q": bit})
    record = child.finalize({"x": child.join(bits, dtype)})
    segments = (("x", width),) if left_only else (("x", width), (None, 1))
    layout = AutoPartition((("word", segments),), left_only=left_only)
    side = RegisterSide.RIGHT if left_only else RegisterSide.THRU
    outer = Register("word", unsigned_register_type(outer_width), side=side)
    program = auto_partition_component_program(record, layout, packed_ports=(outer,))
    packed = next(f for f in program.functions if f.name == "packed")
    assert packed.registers == tuple(
        auto_partition_signature(
            Signature((Register("x", dtype),)), layout, packed_ports=(outer,)
        )
    )
    wires = tuple(range(outer_width))
    bound = bind_register_program(program, (wires,), entry="packed")
    assert bound.outputs == (wires,)
    assert bound.events == ()

    source = Original(
        BitwiseNot(QUInt(width)),
        [
            (
                SourceRegister("word", QUInt(outer_width), side=Side[side.name]),
                ["x"] if left_only else ["x", Unused(1)],
            )
        ],
        left_only=left_only,
    )
    original = source.decompose_bloq()
    cirq_circuit = original.to_cirq_circuit(
        qubit_manager=cirq.SimpleQubitManager(),
        cirq_quregs={"word": np.asarray(cirq.LineQubit.range(outer_width))},
    )
    gates = cirq.decompose(cirq_circuit, keep=lambda op: op.gate == cirq.X)
    assert bound.circuit.instructions == tuple(
        Instruction("x", (gate.qubits[0].x,)) for gate in gates
    )
    assert len(gates) == width
    assert get_cost_value(source, QubitCount()) == outer_width
    values = torch.arange(1, 2**outer_width + 1, dtype=torch.float64)
    initial = torch.complex(values, (values % 3) - 1)
    initial /= initial.norm()
    matrix = torch.as_tensor(original.tensor_contract(), dtype=initial.dtype)
    circuit = Circuit(outer_width, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(outer_width, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().flatten(), matrix @ initial, rtol=0, atol=2e-12
    )
