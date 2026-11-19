"""Typed comparison trees preserve the frozen source's physical execution."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.ordering_component import (
    greater_than_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("adjoint", (False, True))
def test_ordering_source_stream_feedback_and_resources(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.comparison import GreaterThan
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    graph = greater_than_program(width, adjoint=adjoint)
    source = GreaterThan(width, width)
    if adjoint:
        source = source.adjoint()
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    ports = {
        "a": tuple(range(width)),
        "b": tuple(range(width, 2 * width)),
        "target": (2 * width,),
    }
    work = tuple(range(2 * width + 1, 5 * width))
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
    )
    assert_source(bound.circuit.instructions, expected)
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement == 2 * width - 1


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_ordering_full_complex_dirty_target_and_every_raw_history(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.comparison import GreaterThan

    logical = 2 * width + 1
    work = 3 * width - 1
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + work), dtype=torch.complex128)
    initial[:: 1 << work] = amplitudes
    source = GreaterThan(width, width)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    if adjoint:
        matrix = matrix.mH
    expected = torch.zeros_like(initial)
    expected[:: 1 << work] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    bound = bind_register_program(
        greater_than_program(width, adjoint=adjoint),
        (tuple(range(width)), tuple(range(width, 2 * width)), (2 * width,)),
        clean=tuple(range(logical + 1, logical + work + 1)),
    )
    for history in product((0, 1), repeat=2 * width - 1):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** ((2 * width - 1) / 2),
            rtol=0,
            atol=3e-12,
        )
