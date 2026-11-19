"""The grouped predicate retains source call boundaries and ordered connections."""

import pytest

from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.conjunction_program import (
    conjunction_program,
)
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import native_controls
from tests.unit.test_register_control_predicate import (
    CASES,
    ControlCase,
    control_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("uncompute", [False, True])
def test_predicate_root_connections_match_unmodified_source(
    case: ControlCase, uncompute: bool
) -> None:
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    spec, patterns, _ = control_fixture(case)
    original = CtrlSpecAnd(spec)
    controls = native_controls(case, tuple(r.name for r in original.control_registers))
    record = control_predicate_program(controls, patterns, uncompute=uncompute)
    source = original.adjoint() if uncompute else original
    signature = Signature(record.functions[0].registers or ())
    assert _native_trace(record, "predicate") == _source_trace(
        _source(signature), source.decompose_bloq()
    )


@pytest.mark.parametrize("values", [(0, 1, 0), (1, 0, 1, 0)])
@pytest.mark.parametrize("uncompute", [False, True])
def test_ladder_connections_match_unmodified_source(
    values: tuple[int, ...], uncompute: bool
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd

    record = conjunction_program(values, uncompute=uncompute)
    source = MultiAnd(values)
    if uncompute:
        source = source.adjoint()
    signature = Signature(record.functions[0].registers or ())
    assert _native_trace(record, "multi_and") == _source_trace(
        _source(signature), source.decompose_bloq()
    )
