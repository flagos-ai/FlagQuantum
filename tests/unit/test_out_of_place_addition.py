"""Full/truncated sums and measured source erasure on coherent input pairs."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.out_of_place_addition import out_of_place_add_ir
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("msb", [False, True])
def test_adder_compute_erase_source_gates_and_model(n: int, msb: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QUInt, Signature
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bb, inputs = BloqBuilder.from_signature(
        Signature.build_from_dtypes(a=QUInt(n), b=QUInt(n))
    )
    op = OutOfPlaceAdder(n, include_most_significant_bit=msb)
    a, b, c = bb.add(op, **inputs)
    a, b = bb.add(op.adjoint(), a=a, b=b, c=c)
    source = bb.finalize(a=a, b=b)
    left = tuple(range(n))
    right = tuple(range(n, 2 * n))
    out = tuple(range(2 * n, 3 * n + int(msb)))
    bits = tuple(range(40, 40 + len(out) - 1))
    compute = out_of_place_add_ir(left, right, out, include_msb=msb)
    erase = out_of_place_add_ir(
        left, right, out, include_msb=msb, adjoint=True, classical_bits=bits
    )
    expected = reference_arithmetic_gates(
        source,
        {"a": left, "b": right},
        out,
        dict(zip(out[::-1][1:], bits, strict=True)),
        cirq_style=True,
    )
    assert_source((*compute, *erase), expected)
    counts = Counter(g.name for g in (*compute, *erase))
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * len(bits)
    assert counts["measure"] == model.measurement == len(bits)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("msb", [False, True])
def test_sum_complex_amplitudes_and_measured_cleanup(n: int, msb: bool) -> None:
    left = tuple(range(n))
    right = tuple(range(n, 2 * n))
    out = tuple(range(2 * n, 3 * n + int(msb)))
    count = len(out)
    bits = tuple(range(40, 40 + count - 1))
    compute = out_of_place_add_ir(left, right, out, include_msb=msb)
    erase = out_of_place_add_ir(
        left, right, out, include_msb=msb, adjoint=True, classical_bits=bits
    )
    data = torch.arange(1 << (2 * n), dtype=torch.int64)
    total = ((data >> n) + (data & ((1 << n) - 1))) % (1 << count)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << (2 * n + count), dtype=torch.complex128)
    state[data << count] = amps
    expected = torch.zeros_like(state)
    expected[(data << count) | total] = amps
    circuit = Circuit(2 * n + count, inputs=state, dtype=torch.complex128).append(
        FieldOperation(2 * n + count, compute)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    dynamic = DynamicCircuit(
        2 * n + count, inputs=state, dtype=torch.complex128
    ).append(FieldOperation(2 * n + count, (*compute, *erase)))
    result = run_dynamic(dynamic, shots=128, seed=41, strategy="trajectory")
    if bits:
        assert len(
            set(map(tuple, result.classical_bits[:, 40 : 40 + len(bits)].tolist()))
        ) == 1 << len(bits)
    torch.testing.assert_close(
        result.final_states.reshape(128, -1), state.expand(128, -1), rtol=0, atol=1e-12
    )


def test_adder_validation() -> None:
    with pytest.raises(ValueError, match="boolean"):
        out_of_place_add_ir((0,), (1,), (2, 3), include_msb=1)
    with pytest.raises(ValueError, match="output width"):
        out_of_place_add_ir((0,), (1,), (2,))
    with pytest.raises(ValueError, match="distinct"):
        out_of_place_add_ir((0,), (0,), (2, 3))
    with pytest.raises(ValueError, match="classical"):
        out_of_place_add_ir((0,), (1,), (2, 3), adjoint=True)
