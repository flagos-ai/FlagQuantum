"""Unmodified source classical Parity contract before executable admission."""

from __future__ import annotations

from itertools import product

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_source_parity_preserves_inputs_and_returns_exact_bit_xor(count: int) -> None:
    from qualtran._infra.data_types import CBit
    from qualtran._infra.registers import Side
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

    op = Parity(count)
    assert [(r.name, r.dtype, r.shape, r.side) for r in op.signature] == [
        ("x", CBit(), (count,), Side.THRU),
        ("parity", CBit(), (), Side.RIGHT),
    ]
    for values in product((0, 1), repeat=count):
        data = np.array(values, dtype=np.uint8)
        result = op.on_classical_vals(x=data)
        assert result["x"] is data
        assert result["parity"] == sum(values) % 2
        returned, parity = op.call_classically(x=data)
        np.testing.assert_array_equal(returned, data)
        assert parity == sum(values) % 2


@pytest.mark.parametrize("count", [0, 1, 3])
def test_source_parity_lacks_decomposition_and_qec_model(count: int) -> None:
    from qualtran._infra.bloq import DecomposeNotImplementedError
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    op = Parity(count)
    with pytest.raises(DecomposeNotImplementedError):
        op.decompose_bloq()
    with pytest.raises(DecomposeNotImplementedError):
        get_cost_value(op, QECGatesCost())
    assert get_cost_value(op, QubitCount()) == 0
