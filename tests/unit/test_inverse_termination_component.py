"""Stored inverse termination calls retain exact source primitive and feedback streams."""

import pytest

from flagquantum.compiler.registers.modular.inversion.inverse_termination_component import (
    inverse_termination_program,
)
from tests.unit.inverse_component_checks import (
    bind_inverse_component,
    check_inverse_source,
)
from tests.unit.inverse_flag_checks import check_flag_channel
from tests.unit.measurement_checks import certified_and_channel_map

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_stream_operands_feedback_and_models(width: int, adjoint: bool) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep1

    source = _KaliskiIterationStep1(width)
    if adjoint:
        source = source.adjoint()
    check_inverse_source(inverse_termination_program(width, adjoint=adjoint), source)


@pytest.mark.parametrize("width", [2, 3])
def test_all_logical_columns_complex_branches_and_restore(width: int) -> None:
    bound, _, data_width = bind_inverse_component(inverse_termination_program(width))
    inverse, _, _ = bind_inverse_component(
        inverse_termination_program(width, adjoint=True)
    )
    work_width = (
        max(
            w
            for g in (*bound.circuit.instructions, *inverse.circuit.instructions)
            for w in g.wires
        )
        + 1
        - data_width
    )
    count = 1 << data_width
    outputs = []
    for raw in range(count):
        v, m, f, t = raw >> 3, (raw >> 2) & 1, (raw >> 1) & 1, raw & 1
        m ^= f & int(v == 0)
        outputs.append((v << 3) | (m << 2) | ((f ^ m) << 1) | (t ^ m))
    before = tuple(
        tuple(
            (raw << work_width) >> (data_width + work_width - 1 - i) & 1
            for i in range(data_width + work_width)
        )
        for raw in range(count)
    )
    _, events = certified_and_channel_map(bound.circuit.instructions, before)
    check_flag_channel(
        bound.circuit.instructions,
        inverse.circuit.instructions,
        tuple(outputs),
        data_width,
        work_width,
        events,
        raw_branches=True,
    )
