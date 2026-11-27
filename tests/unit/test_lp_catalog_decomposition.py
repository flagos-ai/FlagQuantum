"""The LP catalogue displays the actual source graph rather than a port stub."""

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms import rotations_ui_export as catalog
from examples.algorithms.qpe_window_decomposition import window_program, window_score
from flagquantum.core.ir.program import verify_program_record
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
