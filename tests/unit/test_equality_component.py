"""Integer equality views retain source calls and measured feedback."""

import pytest

from flagquantum.compiler.registers.comparison.equality_component import (
    equality_constant_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,value", [(1, 0), (1, 1), (2, 0), (2, 3), (3, 5), (5, 17), (8, 131)]
)
def test_equality_views_and_source_stream(width: int, value: int) -> None:
    from qualtran.bloqs.arithmetic.comparison import EqualsAConstant

    graph = equality_constant_program(width, value)
    source = EqualsAConstant(width, value)
    compare_layers(graph, source, atomic_leaves(graph))
    ports = {"x": tuple(range(width)), "target": (width,)}
    clean = tuple(range(width + 1, 2 * width))
    bound = bind_register_program(graph, tuple(ports.values()), clean=clean)
    slots = MeasurementSlots(bound.circuit.instructions)
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(source, ports, clean, slots),
    )
