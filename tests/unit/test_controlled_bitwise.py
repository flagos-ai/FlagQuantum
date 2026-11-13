"""Controlled XorK preserves pinned gates and both coherent control polarities."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.bits.bitwise import controlled_xor_constant_ir
from flagquantum.core.ir import Instruction
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def controlled_gates(n: int, k: int, cv: int) -> tuple[Instruction, ...]:
    gates = controlled_xor_constant_ir(0, tuple(range(1, n + 1)), k)
    flip = (Instruction("x", (0,)),) if cv == 0 else ()
    return (*flip, *gates, *flip)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("kind", ["zero", "one", "all", "alternating"])
@pytest.mark.parametrize("cv", [0, 1])
def test_controlled_xork_source_and_model(n: int, kind: str, cv: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    k = {"zero": 0, "one": 1, "all": (1 << n) - 1, "alternating": ((1 << n) - 1) // 3}[
        kind
    ]
    target = tuple(range(1, n + 1))
    source = XorK(QUInt(n), k).controlled(CtrlSpec(cvs=cv))
    gates = controlled_gates(n, k, cv)
    expected = reference_arithmetic_gates(source, {"ctrl": (0,), "x": target}, (), {})
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter({"cx": k.bit_count(), "x": 2 * (cv == 0)})
    assert Counter(g.name for g in gates) == +counts
    assert model.clifford == sum(counts.values())
    assert model.total_t_count() == model.measurement == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["zero", "all", "alternating"])
@pytest.mark.parametrize("cv", [0, 1])
def test_coherent_control_target_spectator_and_double_restore(
    n: int, kind: str, cv: int
) -> None:
    k = {"zero": 0, "all": (1 << n) - 1, "alternating": ((1 << n) - 1) // 3}[kind]
    width = n + 2
    data = torch.arange(1 << width, dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    control = data >> (n + 1)
    mapped = data ^ (((control == cv).to(torch.int64) * k) << 1)
    expected = torch.zeros_like(amps)
    expected[mapped] = amps
    op = FieldOperation(width, controlled_gates(n, k, cv))
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
