"""Stored inverse parity calls retain exact source primitive and feedback streams."""

import pytest

from flagquantum.compiler.registers.modular.inversion.inverse_parity_component import (
    inverse_parity_program,
)
from tests.unit.inverse_component_checks import (
    bind_inverse_component,
    check_inverse_source,
)
from tests.unit.inverse_flag_checks import check_flag_channel
from tests.unit.measurement_checks import certified_and_channel_map

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_stream_operands_feedback_and_models(width: int, adjoint: bool) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep2

    source = _KaliskiIterationStep2(width)
    if adjoint:
        source = source.adjoint()
    check_inverse_source(inverse_parity_program(width, adjoint=adjoint), source)


@pytest.mark.parametrize("width", [1, 2])
def test_all_logical_columns_complex_branches_and_restore(width: int) -> None:
    bound, _, data_width = bind_inverse_component(inverse_parity_program(width))
    inverse, _, _ = bind_inverse_component(inverse_parity_program(width, adjoint=True))
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
    mask = (1 << width) - 1
    outputs = []
    for raw in range(count):
        u, v = raw >> (width + 4), (raw >> 4) & mask
        b, a, m, f = (raw >> 3) & 1, (raw >> 2) & 1, (raw >> 1) & 1, raw & 1
        a ^= f & int(u % 2 == 0)
        m ^= f & int(v % 2 == 0) & (a ^ 1)
        b ^= a ^ m
        outputs.append(
            (u << (width + 4)) | (v << 4) | (b << 3) | (a << 2) | (m << 1) | f
        )
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
