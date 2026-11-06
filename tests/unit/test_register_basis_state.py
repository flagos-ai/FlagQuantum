"""Check the exact X/Z state leaves and their pinned QEC accounting."""

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler.registers.basis_state import (
    Basis,
    basis_effect_cost_key,
    basis_state_cost_key,
    basis_state_ir,
)
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("wire", [0, 1])
def test_basis_state_source_emission_and_full_complex_output(
    basis: Basis, bit: int, wire: int
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran import Side
    from qualtran.bloqs.basic_gates import MinusState, OneState, PlusState, ZeroState
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = ((PlusState, MinusState) if basis == "x" else (ZeroState, OneState))[bit]()
    (port,) = source.signature
    assert port.side == Side.RIGHT and port.name == "q"
    operation, outputs = source.as_cirq_op(qubit_manager=cirq.SimpleQubitManager())
    (allocated,) = outputs["q"]
    leaves = (
        ()
        if operation is None
        else (
            tuple(operation.mapped_circuit().all_operations())
            if isinstance(operation, cirq.CircuitOperation)
            else (operation,)
        )
    )
    assert all(leaf.qubits == (allocated,) for leaf in leaves)
    reference = tuple(
        Instruction({cirq.X: "x", cirq.H: "h"}[leaf.gate], (wire,)) for leaf in leaves
    )
    gates = basis_state_ir(basis, bit, wire)
    assert gates == reference

    state = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(2)
    spectator = torch.tensor([1j, 2 - 1j], dtype=torch.complex128)
    spectator /= spectator.norm()
    clean = torch.tensor([1, 0], dtype=torch.complex128)
    initial = (
        torch.kron(clean, spectator) if wire == 0 else torch.kron(spectator, clean)
    )
    expected = (
        torch.kron(state, spectator) if wire == 0 else torch.kron(spectator, state)
    )
    circuit = fq.Circuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)

    key = basis_state_cost_key(basis, bit)
    assert query_qec_gate_cost(key) == QECGateCost()
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == 1


def test_basis_state_rejects_invalid_bit_and_wire() -> None:
    assert basis_state_ir("z", 0, 0) == ()
    with pytest.raises(ValueError, match="binary bit"):
        basis_state_ir("x", 2, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        basis_state_ir("z", 0, -1)
    with pytest.raises(ValueError, match="binary bit"):
        basis_state_cost_key("x", True)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("bit", [0, 1])
def test_basis_effect_cost_and_source_left_boundary(basis: Basis, bit: int) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran import Side
    from qualtran.bloqs.basic_gates import (
        MinusEffect,
        OneEffect,
        PlusEffect,
        ZeroEffect,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = ((PlusEffect, MinusEffect) if basis == "x" else (ZeroEffect, OneEffect))[
        bit
    ]()
    (port,) = source.signature
    assert port.side == Side.LEFT and port.name == "q"
    with pytest.raises(ValueError, match="no Cirq equivalent"):
        source.as_cirq_op(qubit_manager=cirq.SimpleQubitManager())

    key = basis_effect_cost_key(basis, bit)
    assert key.opcode == f"{basis}_effect"
    assert key.parameters == (bit,)
    assert query_qec_gate_cost(key) == QECGateCost()
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == 1
