"""Check source OnEach wiring, full complex action and modeled multiplicity."""

from dataclasses import asdict
from typing import Any, Literal

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    OnEachCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
)
from flagquantum.compiler.registers.bitwise import bitwise_not_ir, on_each_ir
from flagquantum.core.ir import Instruction
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("opcode", ["x", "h", "rz"])
@pytest.mark.parametrize("dtype", ["raw", "unsigned"])
def test_on_each_source_order_and_entangled_amplitudes(
    opcode: Literal["x", "h", "rz"], dtype: str
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.basic_gates import Hadamard, Rz, XGate
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    angle = 0.37
    leaf = {"x": XGate(), "h": Hadamard(), "rz": Rz(angle)}[opcode]
    wires = (3, 0, 2)
    source = OnEach(3, leaf, QUInt(3) if dtype == "unsigned" else None)
    body = Instruction(opcode, (1,), {"theta": angle} if opcode == "rz" else {})
    gates = on_each_ir(body, wires)

    def source_leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if bloq != leaf:
            return None
        params = {"theta": bloq.angle} if isinstance(bloq, Rz) else {}
        return (Instruction(opcode, inputs["q"], params),), inputs

    expected, outputs = walk_source_graph(
        source, {"q": wires}, {"q": wires}, source_leaf
    )
    assert outputs == {"q": wires}
    assert gates == expected
    assert all(g.wires == (wire,) for g, wire in zip(gates, wires, strict=True))
    assert all(g.params is not body.params for g in gates)
    if opcode == "x":
        assert bitwise_not_ir(wires) == gates

    key = RotationCall("rz", angle) if opcode == "rz" else CliffordCall(opcode, 1)
    modeled = query_qec_gate_cost(OnEachCall(key, len(wires)))
    assert {name: value for name, value in asdict(modeled).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == len(wires)

    state = torch.tensor(
        [complex(i + 1, i % 3 - 1) for i in range(16)], dtype=torch.complex128
    )
    state /= state.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*wires, 1)
    inverse = tuple(order.index(wire) for wire in range(4))
    expected_state = (
        (matrix @ state.reshape(2, 2, 2, 2).permute(order).reshape(8, 2))
        .reshape(2, 2, 2, 2)
        .permute(inverse)
        .reshape(-1)
    )
    circuit = fq.Circuit(4, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(4, gates))
    torch.testing.assert_close(
        circuit.state().flatten(), expected_state, rtol=0, atol=2e-12
    )


@pytest.mark.ftqc_reference
def test_zero_and_symbolic_on_each_resource_counts() -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    count = Symbol("width", positive=True, integer=True)
    key = CliffordCall("x", 1)
    for width in (0, count):
        source = OnEach(width, XGate())
        call = OnEachCall(key, width)
        assert count_selected_calls(call, (key,)) == {key: width}
        assert (
            query_qec_gate_cost(call).clifford
            == get_cost_value(source, QECGatesCost()).clifford
        )
        assert get_cost_value(source, QubitCount()) == width
    assert on_each_ir(Instruction("x", (0,)), ()) == ()
    zero = OnEach(0, XGate())
    assert [
        type(item.bloq).__name__ for item, _, _ in zero.decompose_bloq().iter_bloqsoqs()
    ] == ["Split", "Join"]
    with pytest.raises(DecomposeTypeError, match="symbolic bitsize"):
        OnEach(count, XGate()).decompose_bloq()


def test_on_each_rejects_dynamic_or_overlapping_binding() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        bitwise_not_ir(())
    with pytest.raises(ValueError, match="distinct"):
        on_each_ir(Instruction("x", (0,)), (1, 1))
    with pytest.raises(ValueError, match="single-qubit"):
        on_each_ir(Instruction("cx", (0, 1)), (2,))
    with pytest.raises(ValueError, match="unconditioned"):
        on_each_ir(Instruction("reset", (0,), metadata={"is_dynamic": True}), (1,))
    with pytest.raises(ValueError, match="unconditioned"):
        on_each_ir(Instruction("x", (0,), metadata={"conditions": ((0, 1),)}), (1,))
    with pytest.raises(ValueError, match="nonnegative"):
        OnEachCall(CliffordCall("x", 1), -1)
