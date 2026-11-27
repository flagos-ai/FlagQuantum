"""The LP catalogue displays the actual source graph rather than a port stub."""

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms import rotations_ui_export as catalog
from examples.algorithms.component_ui_dependencies import ComponentFunction
from examples.algorithms.qpe_window_decomposition import (
    window_child,
    window_program,
    window_score,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_basic_gate_child_decomposition import _source_score
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 3, 5, 7))
def test_complete_lp_root_score_matches_original_geometry_and_every_glyph(
    width: int,
) -> None:
    # Cirq is needed by the original wire-symbol protocol even for a source graph.
    assert cirq.__version__
    program = window_program("LPResourceState", {"bitsize": width})
    assert program is not None and verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == 15
    actual = window_score("LPResourceState", {"bitsize": width})
    assert actual is not None
    expected = _source_score(LPResourceState(width).decompose_bloq())
    assert _canonical_score(dict(actual)) == _canonical_score(expected)


def test_real_catalogue_uses_complete_native_body_and_keeps_symbolic_refusal() -> None:
    program = catalog.decomposition("lp_resource_state_small")
    assert program is not None and verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == 15
    assert catalog.score("lp_resource_state_small") == window_score(
        "LPResourceState", {"bitsize": 5}
    )
    assert catalog.decomposition("lp_resource_state_symbolic") is None
    interim_nodes = [
        ComponentFunction(
            "lp",
            function.name,
            program,
            "lp_resource_state_small",
            child_presentation=catalog._child_presentation,
        )
        for function in program.functions
        if tuple(port.name for port in function.registers or ()) == ("m", "anc")
    ]
    assert {node.attributes()["operation"] for node in interim_nodes} == {
        "LPRSInterimPrep",
        "LPRSInterimPrep†",
    }


@pytest.mark.parametrize("width", (1, 3, 5, 7))
def test_lp_interim_dependencies_publish_both_directions_and_controlled_rotations(
    width: int,
) -> None:
    program = window_program("LPResourceState", {"bitsize": width})
    assert program is not None
    source = LPResourceState(width).decompose_bloq()
    operations = program.functions[0].blocks[0].operations
    children = {instance.i: instance.bloq for instance, _, _ in source.iter_bloqsoqs()}
    directions = []
    for position, operation in enumerate(operations):
        assert operation.callee is not None
        presentation = window_child(
            program, operation.callee, {"bitsize": width}, operation="LPResourceState"
        )
        if presentation is None:
            continue
        title, record, symbols = presentation
        directions.append(title)
        score = register_score_data(
            component_register_score(record), register_wire_symbol=symbols
        )
        assert _canonical_score(dict(score)) == _canonical_score(
            _source_score(children[position].decompose_bloq())
        )
        assert symbols is not None and verify_program_record(record) == ()
        nested = children[position].decompose_bloq()
        nested_children = {
            instance.i: instance.bloq for instance, _, _ in nested.iter_bloqsoqs()
        }
        rotations = 0
        for index, call in enumerate(record.functions[0].blocks[0].operations):
            assert call.callee is not None
            child = window_child(
                record, call.callee, {"bitsize": width}, operation="LPResourceState"
            )
            if child is None:
                continue
            assert child[0] == ("CRz†" if title.endswith("†") else "CRz")
            score = register_score_data(
                component_register_score(child[1]), register_wire_symbol=child[2]
            )
            assert _canonical_score(dict(score)) == _canonical_score(
                _source_score(nested_children[index].decompose_bloq())
            )
            rotations += 1
        assert rotations == width
    assert directions == ["LPRSInterimPrep", "LPRSInterimPrep†", "LPRSInterimPrep"]
