"""Compare pinned integer states and effects to native load and contraction."""

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler.registers.layout.int_vector import (
    int_effect_bindings,
    int_state_ir,
    int_vector_cost_key,
)
from flagquantum.core.ir import Instruction
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,value", [(1, 0), (1, 1), (3, 2), (5, 17), (8, 173)])
def test_integer_vectors_keep_source_bit_order_and_complex_action(
    width: int, value: int
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran import Side
    from qualtran.bloqs.basic_gates import (
        IntEffect,
        IntState,
        OneEffect,
        OneState,
        ZeroEffect,
        ZeroState,
    )
    from qualtran.bloqs.bookkeeping import ArbitraryClifford, Join, Split
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source_state, source_effect = IntState(value, width), IntEffect(value, width)
    prep_port, effect_port = source_state.signature[0], source_effect.signature[0]
    assert prep_port.side == Side.RIGHT and effect_port.side == Side.LEFT
    ordered_wires = tuple(reversed(range(1, width + 1)))
    prep = source_state.decompose_bloq()
    state_leaves = [item.bloq for item, _, _ in prep.iter_bloqsoqs()]
    assert isinstance(state_leaves[-1], Join)
    assert all(isinstance(leaf, (ZeroState, OneState)) for leaf in state_leaves[:-1])
    source_circuit, outputs = prep.to_cirq_circuit_and_quregs(
        qubit_manager=cirq.SimpleQubitManager()
    )
    active = set()
    for gate in source_circuit.all_operations():
        assert gate.gate == cirq.X
        active.update(gate.qubits)
    assert tuple(wire in active for wire in outputs["val"].flat) == tuple(
        isinstance(leaf, OneState) for leaf in state_leaves[:-1]
    )
    gates = int_state_ir(ordered_wires, value)
    assert gates == tuple(
        Instruction("x", (wire,))
        for wire, leaf in zip(ordered_wires, state_leaves[:-1], strict=True)
        if isinstance(leaf, OneState)
    )

    effect_leaves = [
        item.bloq for item, _, _ in source_effect.decompose_bloq().iter_bloqsoqs()
    ]
    assert isinstance(effect_leaves[0], Split)
    assert all(isinstance(leaf, (ZeroEffect, OneEffect)) for leaf in effect_leaves[1:])
    bindings = int_effect_bindings(ordered_wires, value)
    assert tuple(bindings.items()) == tuple(
        (wire, ("z", int(isinstance(leaf, OneEffect))))
        for wire, leaf in zip(ordered_wires, effect_leaves[1:], strict=True)
    )
    for source in (source_state, source_effect):
        _, sigma = source.call_graph()
        assert sigma == {ArbitraryClifford(width): 1}
        assert get_cost_value(source, QECGatesCost()).asdict() == {"clifford": 1}
        assert get_cost_value(source, QubitCount()) == width
    assert query_qec_gate_cost(int_vector_cost_key(width)) == QECGateCost(clifford=1)

    spectator = torch.tensor([1j, 2 - 1j], dtype=torch.complex128)
    spectator /= spectator.norm()
    source_vector = torch.as_tensor(
        source_state.tensor_contract(), dtype=torch.complex128
    ).reshape(-1)
    by_register = torch.kron(spectator, source_vector)
    permutation = (0, *ordered_wires)
    expected = by_register.reshape((2,) * (width + 1)).permute(permutation).reshape(-1)
    initial = torch.zeros_like(expected)
    initial[:: 1 << width] = spectator
    circuit = fq.Circuit(width + 1, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)

    values = torch.arange(1 << (width + 1), dtype=torch.float64)
    state = torch.complex(values + 1, values % 7 - 2)
    state /= state.norm()
    plan = build_tensor_network(
        fq.Circuit(width + 1, inputs=state, dtype=torch.complex128),
        dtype=torch.complex128,
    )
    bra = torch.as_tensor(
        source_effect.tensor_contract(), dtype=torch.complex128
    ).reshape(-1)
    expected_effect = (
        state.reshape((2,) * (width + 1)).permute(permutation).reshape(2, -1) @ bra
    )
    effect = tensor_network_basis_effect(plan, bindings)
    torch.testing.assert_close(effect[0], expected_effect, rtol=0, atol=2e-12)
    assert float(effect.norm()) < 1


@pytest.mark.ftqc_reference
def test_integer_vector_symbolic_and_zero_width_source_boundaries() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqError, DecomposeTypeError
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    width = Symbol("width", integer=True, positive=True)
    value = Symbol("value", integer=True, nonnegative=True)
    key = int_vector_cost_key(width)
    assert key.width == width
    assert query_qec_gate_cost(key) == QECGateCost(clifford=1)
    for make in (IntState, IntEffect):
        symbolic_width = make(0, width)
        assert get_cost_value(symbolic_width, QubitCount()) == width
        assert get_cost_value(symbolic_width, QECGatesCost()).asdict() == {
            "clifford": 1
        }
        with pytest.raises(DecomposeTypeError, match="Symbolic bitsize"):
            symbolic_width.decompose_bloq()
        symbolic_value = make(value, 3)
        assert get_cost_value(symbolic_value, QECGatesCost()).asdict() == {
            "clifford": 1
        }
        with pytest.raises(ValueError, match="should be an integer"):
            symbolic_value.decompose_bloq()
        with pytest.raises(ValueError, match="too big"):
            make(4, 2)
    with pytest.raises(BloqError, match="dtypes are not consistent"):
        IntState(0, 0).decompose_bloq()
    with pytest.raises(IndexError):
        IntEffect(0, 0).decompose_bloq()
    assert query_qec_gate_cost(int_vector_cost_key(0)) == QECGateCost(clifford=1)


def test_integer_binding_requires_valid_nonempty_register_and_fitting_value() -> None:
    for bind in (int_state_ir, int_effect_bindings):
        with pytest.raises(ValueError, match="nonempty"):
            bind((), 0)
        with pytest.raises(ValueError, match="fit"):
            bind((1, 0), 4)
        with pytest.raises(ValueError, match="fit"):
            bind((1, 0), -1)
        with pytest.raises(ValueError, match="distinct"):
            bind((0, 0), 0)
