"""Cover the complete locked Power test file through native owners."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    PowerCall,
    RotationCall,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_power import power_word_program
from flagquantum.compiler.registers.primitives.power import register_power_ir
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.operator_schema import MAX_POWER_REPEATS
from flagquantum.drawer.power_symbols import power_wire_symbols
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_power_preserves_shaped_signature_and_ten_owned_calls() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Power
    from qualtran.bloqs.for_testing import TestMultiRegister

    signature = Signature(
        (
            Register("xx", quantum_bit_type()),
            Register("yy", opaque_register_type(2), (2, 2)),
            Register("zz", opaque_register_type(3)),
        )
    )
    leaf = bit_gate_function(
        "x", tuple(Signature.build(q=1)), CircuitIR(1, (Instruction("x", (0,)),))
    )
    pair = bit_gate_function(
        "cx",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(signature, (leaf, pair), name="child")
    outputs = dict(builder.inputs)
    outputs["xx"] = builder.call_dict("x", {"q": outputs["xx"]})["q"]
    shaped = np.asarray(outputs["yy"], dtype=object).copy()
    for index in np.ndindex(shaped.shape):
        ctrl, target = builder.split(shaped[index])
        result = builder.call_dict("cx", {"ctrl": ctrl, "target": target})
        shaped[index] = builder.join((result["ctrl"], result["target"]))
    outputs["yy"] = shaped
    record = builder.finalize(outputs)
    child = next(fn for fn in record.functions if fn.name == "child")

    source_child = TestMultiRegister()
    source = Power(source_child, 10)
    assert [(r.name, r.bitsize, r.shape) for r in signature] == [
        (r.name, r.bitsize, r.shape) for r in source.signature
    ]
    assert source.signature == source_child.signature
    original = [
        instance.bloq for instance, _, _ in source.decompose_bloq().iter_bloqnections()
    ]
    assert original == [source_child] * 10

    definitions = tuple(fn for fn in record.functions if fn.name != child.name)
    powered = power_word_program(child, 10, definitions=definitions)
    assert verify_program_record(powered) == ()
    main = next(fn for fn in powered.functions if fn.name == "main")
    assert main.registers == tuple(signature)
    block = main.blocks[0]
    assert [op.callee for op in block.operations] == [child.name] * len(original)
    previous = block.arguments
    for operation in block.operations:
        assert operation.operands == previous
        previous = operation.results
    assert block.return_values == previous

    # The original fixture's TestAtom and TestTwoBitOp tensors are X and CX.
    # The zz split and join has no physical effect.
    wires = ((11,), (9, 10), (7, 8), (5, 6), (3, 4), (0, 1, 2))
    bound = bind_register_program(powered, wires)
    assert bound.outputs == wires and not bound.events
    single_call = (
        Instruction("x", (11,)),
        *(Instruction("cx", pair) for pair in wires[1:-1]),
    )
    assert bound.circuit.instructions == single_call * 10
    directional = replace(
        child, registers=(Register("q", quantum_bit_type(), side=RegisterSide.LEFT),)
    )
    with pytest.raises(ValueError, match="THRU"):
        power_word_program(directional, 0)
    with pytest.raises(ValueError, match="positive integer"):
        power_word_program(child, -2)
    with pytest.raises(ValueError, match="exceeds"):
        power_word_program(child, MAX_POWER_REPEATS + 1)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("inner,outer", [(2, 3), (-2, -3), (2, -3)])
def test_flattened_signed_power_matches_source_exactly(inner: int, outer: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Power, Rz
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    angle = 0.37
    body = CircuitIR(1, (Instruction("rz", (0,), {"theta": angle}),))
    first = register_power_ir(body, abs(inner), adjoint=inner < 0)
    nested = register_power_ir(CircuitIR(1, first), abs(outer), adjoint=outer < 0)
    flat = register_power_ir(body, abs(inner * outer), adjoint=inner * outer < 0)
    source = (Power(Rz(angle), 1) ** inner) ** outer
    assert nested == flat
    assert len(nested) == source.power
    assert [gate.params["theta"] for gate in nested] == [
        item.bloq.angle for item, _, _ in source.decompose_bloq().iter_bloqsoqs()
    ]
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(FieldOperation(1, nested))
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(source.tensor_contract()),
        rtol=0,
        atol=2e-12,
    )
    model = query_qec_gate_cost(
        PowerCall(RotationCall("rz", angle), abs(inner * outer))
    )
    assert model.rotation == get_cost_value(source, QECGatesCost()).rotation


@pytest.mark.ftqc_reference
def test_original_atom_and_shaped_power_labels() -> None:
    pytest.importorskip("qualtran")
    import cirq
    from qualtran.bloqs.basic_gates import Power
    from qualtran.bloqs.for_testing import TestMultiRegister
    from qualtran.bloqs.for_testing.atom import TestGWRAtom
    from sympy import Symbol

    for source, labels in (
        (TestGWRAtom(), ("TestGWRAtom",)),
        (TestMultiRegister(), ("Power", *("yy",) * 8, *("zz",) * 3)),
    ):
        powered = Power(source, 4)
        info = cirq.circuit_diagram_info(powered)
        assert power_wire_symbols(labels, 4) == tuple(info.wire_symbols)
    count = Symbol("n")
    assert power_wire_symbols(("U", "target"), count) == ("U^n", "target^n")


def test_power_owners_import_in_fresh_process() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import flagquantum.compiler.registers.component_power; "
            "import flagquantum.simulation.classical_power; import flagquantum.drawer.power_symbols",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=30,
    )
