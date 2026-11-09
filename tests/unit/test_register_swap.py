"""Verify literal SWAP order and costs against the pinned source."""

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.primitives.controlled_swap import (
    controlled_swap_bit_ir,
    controlled_swap_ir,
)
from flagquantum.compiler.registers.primitives.swap import (
    source_swap_resource_formula,
    swap_bit_ir,
    swap_register_ir,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 8])
def test_register_swap_matches_source_cirq_order_and_resource_counts(
    width: int,
) -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.basic_gates.swap import CSwap, Swap, TwoBitCSwap, TwoBitSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Swap(width)
    assert source.adjoint() == source
    assert source.controlled() == CSwap(width)
    assert TwoBitSwap().controlled() == TwoBitCSwap()
    left = tuple(reversed(range(0, 2 * width, 2)))
    right = tuple(range(1, 2 * width, 2))
    qubits = {
        name: np.array([cirq.LineQubit(wire) for wire in wires])
        for name, wires in (("x", left), ("y", right))
    }
    cirq_circuit, outputs = source.decompose_bloq().to_cirq_circuit_and_quregs(
        qubit_manager=cirq.SimpleQubitManager(), **qubits
    )
    gates = swap_register_ir(left, right)
    actual = tuple(
        cirq.SWAP(*(cirq.LineQubit(q) for q in gate.wires)) for gate in gates
    )
    assert all(gate.name == "swap" for gate in gates)
    assert actual == tuple(cirq_circuit.all_operations())
    assert tuple(q.x for q in outputs["x"]) == left
    assert tuple(q.x for q in outputs["y"]) == right
    assert gates == tuple(
        g for a, b in zip(left, right, strict=True) for g in swap_bit_ir(a, b)
    )

    cost = source_swap_resource_formula(width)
    assert cost.logical_qubits == get_cost_value(source, QubitCount()) == 2 * width
    assert get_cost_value(source, QECGatesCost()).asdict() == {
        "clifford": cost.clifford
    }
    assert len(gates) == cost.clifford and cost.measurement == cost.t == 0
    assert controlled_swap_ir(2 * width, left, right) == tuple(
        gate
        for x, y in zip(left, right, strict=True)
        for gate in controlled_swap_bit_ir(2 * width, x, y)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3])
def test_swap_full_complex_action_and_self_inverse(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.swap import Swap, TwoBitSwap

    spectator = 2 * width
    left = tuple(reversed(range(0, 2 * width, 2)))
    right = tuple(range(1, 2 * width, 2))
    values = torch.arange(1, 1 + (1 << (spectator + 1)), dtype=torch.float64)
    initial = torch.complex(values, values.remainder(7) - 3)
    initial /= initial.norm()
    circuit = fq.Circuit(spectator + 1, inputs=initial, dtype=initial.dtype)
    operation = FieldOperation(spectator + 1, swap_register_ir(left, right))
    circuit.append(operation)
    source = TwoBitSwap() if width == 1 else Swap(width)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=initial.dtype).reshape(
        1 << spectator, 1 << spectator
    )
    order = (spectator, *left, *right)
    shape = (2,) * (spectator + 1)
    expected = initial.reshape(shape).permute(order).reshape(2, -1) @ matrix.T
    actual = circuit.state().reshape(shape).permute(order).reshape(2, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    torch.testing.assert_close(circuit.append(operation).state().reshape(-1), initial)


@pytest.mark.ftqc_reference
def test_swap_symbolic_and_zero_width_boundaries() -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates.swap import Swap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    n = Symbol("n", positive=True, integer=True)
    source = Swap(n)
    cost = source_swap_resource_formula(n)
    assert get_cost_value(source, QubitCount()) == cost.logical_qubits == 2 * n
    assert get_cost_value(source, QECGatesCost()).asdict() == {
        "clifford": cost.clifford
    }
    with pytest.raises(DecomposeTypeError, match="concrete"):
        source.decompose_bloq()
    with pytest.raises(TypeError, match="missing"):
        Swap(0).decompose_bloq()
    assert get_cost_value(Swap(0), QECGatesCost()).asdict() == {}
    with pytest.raises(ValueError, match="nonempty"):
        swap_register_ir((), ())


def test_swap_rejects_mismatched_and_aliased_bindings() -> None:
    with pytest.raises(ValueError, match="equal"):
        swap_register_ir((0,), (1, 2))
    with pytest.raises(ValueError, match="distinct"):
        swap_bit_ir(1, 1)
    with pytest.raises(ValueError, match="positive"):
        source_swap_resource_formula(-1)
