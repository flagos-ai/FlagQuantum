"""Compare register identity binding and zero-cost width with the pinned source."""

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    identity_cost_key,
)
from flagquantum.compiler.registers.identity import register_identity_ir
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,controls", [(0, 0), (0, 2), (1, 0), (3, 0), (3, 2)])
def test_identity_source_width_cost_and_exact_action(width: int, controls: int) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Identity(width)
    if controls:
        source = source.controlled(CtrlSpec(cvs=np.asarray([0, 1])))
    cost = identity_cost_key(width, control_width=controls)
    assert cost.parameters == (source.bitsize,)
    assert source.adjoint() == source
    assert query_qec_gate_cost(cost) == QECGateCost()
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == width + controls

    wires = tuple(reversed(range(width + controls)))
    gates = register_identity_ir(wires[controls:], controls=(wires[:controls],))
    assert gates == tuple(Instruction("i", (wire,)) for wire in wires)
    # Use one extra spectator, including the case with no logical ports.
    size = width + controls + 1
    state = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << size)], dtype=torch.complex128
    )
    state /= state.norm()
    circuit = fq.Circuit(size, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(size, gates))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)
    if width + controls:
        assert np.array_equal(source.tensor_contract(), np.eye(1 << (width + controls)))


@pytest.mark.ftqc_reference
def test_identity_ctrl_adder_preserves_shaped_group_order() -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran import BloqBuilder, CtrlSpec, QBit, QUInt, Register
    from qualtran.bloqs.basic_gates import Identity

    spec = CtrlSpec((QBit(), QUInt(2)), (np.asarray([0, 1]), np.asarray(2)))
    source, add = Identity(2).get_ctrl_system(spec)
    builder = BloqBuilder()
    left = builder.add_register(Register("a", QBit(), shape=(2,)))
    right = builder.add_register(Register("b", QUInt(2)))
    data = builder.add_register(Register("q", QUInt(2)))
    (a, b), (q,) = add(builder, (left, right), {"q": data})
    composite = builder.finalize(a=a, b=b, q=q)
    assert len(tuple(composite.iter_bloqsoqs())) == 3

    wire_groups = {"a": (5, 1), "b": (0, 4), "q": (3, 2)}
    quregs = {
        name: np.asarray([cirq.LineQubit(wire) for wire in wires]).reshape(
            (2, 1) if name == "a" else (2,)
        )
        for name, wires in wire_groups.items()
    }
    circuit, out = composite.to_cirq_circuit_and_quregs(
        qubit_manager=cirq.SimpleQubitManager(), **quregs
    )
    (atomic,) = tuple(circuit.all_operations())
    assert atomic.gate == cirq.IdentityGate(source.bitsize)
    for name, wires in wire_groups.items():
        assert tuple(qubit.x for qubit in out[name].flat) == wires
    assert tuple(
        gate.wires[0]
        for gate in register_identity_ir(
            wire_groups["q"], controls=(wire_groups["a"], wire_groups["b"])
        )
    ) == tuple(qubit.x for qubit in atomic.qubits)


@pytest.mark.ftqc_reference
def test_identity_symbolic_width_preserves_the_opaque_cost_leaf() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.resource_counting import QubitCount, get_cost_value
    from sympy import Symbol

    width = Symbol("width", positive=True, integer=True)
    source = Identity(width).controlled()
    key = identity_cost_key(width, control_width=1)
    assert key.parameters == (source.bitsize,)
    assert get_cost_value(source, QubitCount()) == width + 1
    assert query_qec_gate_cost(key) == QECGateCost()


def test_identity_rejects_aliases_and_negative_widths() -> None:
    with pytest.raises(ValueError, match="distinct"):
        register_identity_ir((0,), controls=((1, 0),))
    with pytest.raises(ValueError, match="nonnegative"):
        identity_cost_key(2, control_width=-1)
