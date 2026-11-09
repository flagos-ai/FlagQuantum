"""Shared positive controlled XorK preserves pinned gates and coherent control."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.bits.bitwise import controlled_xor_constant_ir
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("kind", ["zero", "one", "all", "alternating"])
def test_positive_controlled_xork_source_and_model(n: int, kind: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    k = {"zero": 0, "one": 1, "all": (1 << n) - 1, "alternating": ((1 << n) - 1) // 3}[
        kind
    ]
    target = tuple(range(1, n + 1))
    source = XorK(QUInt(n), k).controlled()
    gates = controlled_xor_constant_ir(0, target, k)
    expected = reference_arithmetic_gates(source, {"ctrl": (0,), "x": target}, (), {})
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    assert Counter(g.name for g in gates) == ({"cx": k.bit_count()} if k else {})
    assert model.clifford == k.bit_count()
    assert model.total_t_count() == model.measurement == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["zero", "all", "alternating"])
def test_coherent_control_target_spectator_and_double_restore(
    n: int, kind: str
) -> None:
    k = {"zero": 0, "all": (1 << n) - 1, "alternating": ((1 << n) - 1) // 3}[kind]
    width = n + 2
    data = torch.arange(1 << width, dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    control = data >> (n + 1)
    mapped = data ^ ((control * k) << 1)
    expected = torch.zeros_like(amps)
    expected[mapped] = amps
    op = FieldOperation(width, controlled_xor_constant_ir(0, tuple(range(1, n + 1)), k))
    circuit = Circuit(width, inputs=amps, dtype=torch.complex128).append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        circuit.append(op).state().flatten(), amps, rtol=0, atol=1e-12
    )


def test_controlled_bitwise_validation() -> None:
    for control in (0, -1, True):
        with pytest.raises(ValueError, match="control"):
            controlled_xor_constant_ir(control, (0, 1), 0)
    with pytest.raises(ValueError, match="unsigned"):
        controlled_xor_constant_ir(2, (0, 1), -1)
