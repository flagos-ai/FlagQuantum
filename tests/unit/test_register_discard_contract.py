"""Unchanged source classical Discard signature, tensor and resource contract."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("bit", [0, 1])
def test_source_discard_consumes_one_classical_bit(bit: int) -> None:
    from qualtran import CBit, Side
    from qualtran.bloqs.basic_gates.discard import Discard
    from qualtran.simulation.tensor import DiscardInd

    op = Discard()
    assert [(r.name, r.dtype, r.shape, r.side) for r in op.signature] == [
        ("c", CBit(), (), Side.LEFT)
    ]
    assert op.on_classical_vals(c=bit) == {}
    assert op.call_classically(c=bit) == ()
    assert op.my_tensors({"c": "input"}, {}) == [DiscardInd(("input", 0))]


def test_source_discard_has_zero_quantum_width_and_no_decomposition() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.basic_gates.discard import Discard
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert get_cost_value(Discard(), QubitCount()) == 0
    with pytest.raises(DecomposeNotImplementedError):
        Discard().decompose_bloq()
    cost = get_cost_value(Discard(), QECGatesCost())
    assert cost.total_t_count() == cost.measurement == cost.clifford == 0
