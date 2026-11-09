"""Pinned register permutations preserve complete coherent phases without work."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.bits.bitwise import (
    bitwise_not_ir,
    xor_constant_ir,
    xor_ir,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("kind", ["xor", "constant", "not"])
def test_bitwise_source_gates_and_model(n: int, dtype: str, kind: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    x, y = tuple(range(n)), tuple(range(n, 2 * n))
    inputs = {"x": x}
    if kind == "xor":
        source = Xor(cls(n))
        gates = xor_ir(x, y)
        inputs["y"] = y
    elif kind == "constant":
        # The emitter takes a bit pattern; signed dtype encoding stays upstream.
        k = -1 if dtype == "signed" else ((1 << n) - 1) // 3
        source = XorK(cls(n), k)
        gates = xor_constant_ir(x, k % (1 << n))
    else:
        source = BitwiseNot(cls(n))
        gates = bitwise_not_ir(x)
    expected = reference_arithmetic_gates(source, inputs, (), {})
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == model.measurement == 0
    assert model.clifford == len(gates) == sum(counts.values())


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["xor", "constant", "not"])
def test_complete_complex_permutation_and_involution(n: int, kind: str) -> None:
    x, y = tuple(range(n)), tuple(range(n, 2 * n))
    width = (2 * n if kind == "xor" else n) + 1  # Last wire is an entangled spectator.
    data = torch.arange(1 << width, dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    if kind == "xor":
        gates = xor_ir(x, y)
        mapped = data ^ ((data >> (n + 1)) << 1)
    else:
        k = ((1 << n) - 1) // 3 if kind == "constant" else (1 << n) - 1
        gates = xor_constant_ir(x, k) if kind == "constant" else bitwise_not_ir(x)
        mapped = data ^ (k << 1)
    expected = torch.zeros_like(amps)
    expected[mapped] = amps
    op = FieldOperation(width, gates)
    circuit = Circuit(width, inputs=amps, dtype=torch.complex128).append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        circuit.append(op).state().flatten(), amps, rtol=0, atol=1e-12
    )
    assert {g.name for g in gates} <= {"x", "cx"}


def test_bitwise_validation() -> None:
    with pytest.raises(ValueError, match="equal nonempty"):
        xor_ir((), ())
    with pytest.raises(ValueError, match="equal nonempty"):
        xor_ir((0,), (1, 2))
    with pytest.raises(ValueError, match="distinct"):
        xor_ir((0,), (0,))
    with pytest.raises(ValueError, match="nonempty"):
        bitwise_not_ir(())
    for wires in ((-1,), (True,), (1, 1)):
        with pytest.raises(ValueError, match="distinct"):
            bitwise_not_ir(wires)
    for value in (-1, 4, True):
        with pytest.raises(ValueError, match="unsigned"):
            xor_constant_ir((0, 1), value)
