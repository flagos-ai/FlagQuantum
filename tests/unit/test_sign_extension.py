"""Shared sign copy matches the pinned extension/truncation pair."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.sign_extension import sign_extend_ir
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,extra", [(1, 1), (2, 1), (3, 2), (3, 5), (5, 3), (8, 8)])
@pytest.mark.parametrize("ones", [False, True])
def test_sign_extension_source_pair_and_complex_outputs(
    n: int, extra: int, ones: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QInt, QIntOnesComp, Signature
    from qualtran.bloqs.arithmetic.conversions.sign_extension import SignExtend

    if ones and n == 1:
        with pytest.raises(ValueError, match="bitsize"):
            QIntOnesComp(n)
        return
    dtype = QIntOnesComp if ones else QInt
    bb, inputs = BloqBuilder.from_signature(Signature.build_from_dtypes(x=dtype(n)))
    op = SignExtend(dtype(n), dtype(n + extra))
    value = bb.add(op, x=inputs["x"])
    value = bb.add(op.adjoint(), x=value)
    source = bb.finalize(x=value)
    x = tuple(range(n))
    prefix = tuple(range(n, n + extra))
    gates = sign_extend_ir(x, prefix)
    expected = reference_arithmetic_gates(source, {"x": x}, prefix, {})
    assert_source((*gates, *gates), expected)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts == {"cx": 2 * extra - 1}
    assert model.total_t_count() == model.measurement == 0

    state = torch.zeros(1 << (n + extra), dtype=torch.complex128)
    values = torch.arange(1 << n, dtype=torch.int64)
    coeff = values.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state[values << extra] = amps
    mapped = (values << extra) | (((values >> (n - 1)) & 1) * ((1 << extra) - 1))
    expected_state = torch.zeros_like(state)
    expected_state[mapped] = amps
    circuit = Circuit(n + extra, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n + extra, gates)
    )
    torch.testing.assert_close(
        circuit.state().flatten(), expected_state, rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        circuit.append(FieldOperation(n + extra, gates)).state().flatten(),
        state,
        rtol=0,
        atol=1e-12,
    )


def test_sign_validation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        sign_extend_ir((0,), ())
    with pytest.raises(ValueError, match="distinct"):
        sign_extend_ir((0,), (0,))
