"""Exact upstream Fredkin primitives, all columns, coherent register exchange."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.controlled_swap import (
    controlled_swap_bit_ir,
    controlled_swap_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("value", range(8))
def test_all_literal_columns_have_exact_positive_phase(value: int) -> None:
    state = torch.zeros(8, dtype=torch.complex128)
    state[value] = 1
    control, left, right = (value >> 2) & 1, (value >> 1) & 1, value & 1
    out = (
        (control << 2)
        | ((right if control else left) << 1)
        | (left if control else right)
    )
    expected = torch.zeros_like(state)
    expected[out] = 1
    gates = controlled_swap_bit_ir(0, 1, 2)
    actual = (
        Circuit(3, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(3, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    mapped, events = certified_and_channel_map(gates, ((control, left, right),))
    assert mapped == (((out >> 2) & 1, (out >> 1) & 1, out & 1),) and events == 0
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap

    upstream = cirq.unitary(TwoBitCSwap().to_clifford_t_circuit())
    # Named-qubit lexical order is ctrl,x,y.
    np.testing.assert_allclose(upstream[:, value], expected.numpy(), rtol=0, atol=1e-12)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_original_register_gate_order_and_independent_resources(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b = tuple(range(1, n + 1)), tuple(range(n + 1, 2 * n + 1))
    source = CSwap(n)
    expected = reference_arithmetic_gates(
        source, {"ctrl": (0,), "x": a, "y": b}, (), {}, preserve_components=True
    )
    actual = controlled_swap_ir(0, a, b)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7) == 7 * n
    assert model.total_t_count() == 4 * n
    assert counts["cx"] == 8 * n and counts["h"] == 2 * n
    assert counts["measure"] == model.measurement == 0
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_all_complex_input_amplitudes_with_entangled_spectator_and_involution(
    n: int,
) -> None:
    width = 2 + 2 * n
    data = torch.arange(1 << width)
    c, x, y, s = (
        data >> (2 * n + 1),
        (data >> (n + 1)) & ((1 << n) - 1),
        (data >> 1) & ((1 << n) - 1),
        data & 1,
    )
    mapped = (
        (c << (2 * n + 1))
        | (torch.where(c == 1, y, x) << (n + 1))
        | (torch.where(c == 1, x, y) << 1)
        | s
    )
    coeff = data.to(torch.float64)
    state = torch.complex(coeff + 1, 3 * coeff - 5)
    state /= state.norm()
    expected = torch.zeros_like(state)
    expected[mapped] = state
    gates = controlled_swap_ir(
        0, tuple(range(1, n + 1)), tuple(range(n + 1, 2 * n + 1))
    )
    circuit = Circuit(width, inputs=state, dtype=torch.complex128).append(
        FieldOperation(width, gates)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        circuit.append(FieldOperation(width, gates)).state().flatten(),
        state,
        rtol=0,
        atol=1e-12,
    )


def test_validation_and_certificate_rejects_incomplete_unitary_block() -> None:
    for a, b in [((), ()), ((1,), (2, 3))]:
        with pytest.raises(ValueError, match="equal nonempty"):
            controlled_swap_ir(0, a, b)
    with pytest.raises(ValueError, match="distinct"):
        controlled_swap_bit_ir(0, 0, 1)
    with pytest.raises(ValueError, match="distinct"):
        controlled_swap_ir(0, (1,), (1,))
    with pytest.raises(AssertionError):
        certified_and_channel_map(controlled_swap_bit_ir(0, 1, 2)[:-1], ((1, 0, 1),))
