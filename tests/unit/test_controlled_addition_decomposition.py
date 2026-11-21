"""Complete controlled-addition catalog source calls, glyphs and large boundary."""

from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.controlled_addition_decomposition import (
    controlled_addition_catalog_program,
    controlled_addition_score,
)
from flagquantum.compiler.registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", ("cadd_small", "cadd_large"))
def test_complete_actual_factory_order_and_immutable_source_drawing_boundary(
    originals: tuple[Any, dict[str, Any]], identifier: str
) -> None:
    from qualtran.drawing import get_musical_score_data

    ports, parameters = native.signature(identifier), native.attributes(identifier)
    record = controlled_addition_catalog_program(
        native.name(identifier), ports, parameters
    )
    assert record is not None and verify_program_record(record) == ()
    assert record == controlled_addition_program(ports[1].dtype, ports[2].dtype)
    source = originals[1][identifier].decompose_bloq()
    assert _native_trace(record) == _source_trace(_source(ports), source)
    actual = controlled_addition_score(native.name(identifier), ports, parameters)
    assert actual is not None
    if identifier == "cadd_large":
        assert len(record.functions[0].blocks[0].operations) == 2008
        with pytest.raises(IndexError):
            get_musical_score_data(source)
        return
    fresh = get_musical_score_data(source)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    assert _canonical_score(dict(actual)) == _canonical_score(expected)


@pytest.mark.parametrize("polarity", (0, 1))
def test_both_control_polarities_retain_original_child_glyphs(polarity: int) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.drawing import get_musical_score_data

    ports = native.signature("cadd_small")
    record = controlled_addition_catalog_program("CAdd", ports, {"cv": polarity})
    assert record == controlled_addition_program(ports[1].dtype, control_value=polarity)
    source = CAdd(QUInt(3), cv=polarity).decompose_bloq()
    assert record is not None
    assert _native_trace(record) == _source_trace(_source(ports), source)
    fresh = get_musical_score_data(source)
    expected = {
        "max_x": fresh.max_x,
        "max_y": fresh.max_y,
        **{
            name: [item.json_dict() for item in getattr(fresh, name)]
            for name in ("soqs", "hlines", "vlines")
        },
    }
    actual = controlled_addition_score("CAdd", ports, {"cv": polarity})
    assert actual is not None
    assert _canonical_score(dict(actual)) == _canonical_score(expected)
