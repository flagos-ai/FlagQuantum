"""Keep actual polynomial factory protocol and bookkeeping count distinctions."""

from collections import Counter

import pytest

from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("constant", (False, True))
@pytest.mark.parametrize("symbolic", (False, True))
def test_actual_factory_protocols(constant: bool, symbolic: bool) -> None:
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import (
        _gf2_poly_4_8_add,
        _gf2_poly_add_symbolic,
    )
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add_k import (
        _gf2_poly_4_8_add_k,
        _gf2_poly_add_k_symbolic,
    )

    pair = (
        (_gf2_poly_4_8_add_k, _gf2_poly_add_k_symbolic)
        if constant
        else (_gf2_poly_4_8_add, _gf2_poly_add_symbolic)
    )
    observed = source_example_protocols(pair[int(symbolic)])
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": "NA" if symbolic else "PASS",
        "counts": "UNVERIFIED" if symbolic else "PASS" if constant else "FAIL",
        "serialize": "FAIL",
        "qtyping": "NA" if symbolic else "PASS",
    }
    assert "data_types.QGFPoly" in observed["serialize"][1]
    if not symbolic and not constant:
        assert "GFPolySplit" in observed["counts"][1]
        assert "GFPolyJoin" in observed["counts"][1]


def test_source_count_failure_is_only_the_declared_view_nodes() -> None:
    from qualtran.bloqs.gf_arithmetic import GF2Addition
    from qualtran.bloqs.gf_poly_arithmetic.gf2_poly_add import _gf2_poly_4_8_add
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
        GFPolyJoin,
        GFPolySplit,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = _gf2_poly_4_8_add.make()
    _, declared = source.call_graph(max_depth=1)
    body = source.decompose_bloq()
    actual = Counter(instance.bloq for instance in body.bloq_instances)
    assert actual == Counter(declared) + Counter(
        {GFPolySplit(source.qgf_poly): 2, GFPolyJoin(source.qgf_poly): 2}
    )
    assert declared == {GF2Addition(source.qgf_poly.qgf): 5}
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == 15
    assert model.total_t_count() == 0
    # Exact direct-call mismatches persist even though both view kinds are
    # gate-free and the independent physical/model Clifford totals agree.
    for view in (GFPolySplit(source.qgf_poly), GFPolyJoin(source.qgf_poly)):
        assert get_cost_value(view, QECGatesCost()).clifford == 0
