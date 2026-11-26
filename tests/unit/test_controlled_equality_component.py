"""Controlled equality retains original gates and every measured branch."""

from collections import Counter
from importlib import import_module

import pytest

from flagquantum.compiler.registers.comparison.controlled_equality_component import (
    controlled_equality_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import montgomery_register_type
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5))
@pytest.mark.parametrize("activation", (0, 1))
def test_original_calls_primitives_feedback_and_independent_model(
    width: int, activation: int
) -> None:
    qt = import_module("qualtran")
    comparison = import_module("qualtran.bloqs.arithmetic.comparison")
    costs = import_module("qualtran.resource_counting")
    graph = controlled_equality_program(
        montgomery_register_type(width), control_value=activation
    )
    child = comparison.Equals(qt.QMontgomeryUInt(width))
    source = (
        child.controlled(ctrl_spec=qt.CtrlSpec(cvs=0))
        if activation == 0
        else import_module("qualtran._infra.controlled").Controlled(
            child, qt.CtrlSpec()
        )
    )
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    inputs = (
        (0,),
        tuple(range(1, width + 1)),
        tuple(range(width + 1, 2 * width + 1)),
        (2 * width + 1,),
    )
    clean = tuple(range(2 * width + 2, 3 * width + 2))
    bound = bind_register_program(graph, inputs, clean=clean)
    expected = reference_arithmetic_gates(
        source,
        dict(zip(("ctrl", "x", "y", "target"), inputs, strict=True)),
        clean,
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected)
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = costs.get_cost_value(source, costs.QECGatesCost())
    assert counts["measure"] == model.measurement == width
    assert counts["ccx"] == model.toffoli == 2 * width
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq == 4 * width
    assert model.total_t_count() == 12 * width
    assert bound.outputs == inputs


@pytest.mark.parametrize("activation", (0, 1))
def test_all_dirty_logical_inputs_complex_amplitudes_all_histories_and_clean_work(
    activation: int,
) -> None:
    width, logical = 3, 8
    graph = controlled_equality_program(
        montgomery_register_type(width), control_value=activation
    )
    bound = bind_register_program(
        graph, ((0,), (1, 2, 3), (4, 5, 6), (7,)), clean=(8, 9, 10)
    )
    before, after = {}, {}
    for word in range(1 << logical):
        control, x, y = word >> 7, (word >> 4) & 7, (word >> 1) & 7
        output = word ^ int(control == activation and x == y)
        before[word << 3] = complex(word + 1, 17 - word)
        after[output << 3] = complex(word + 1, 17 - word)
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before = {key: value / norm for key, value in before.items()}
    after = {key: value / norm for key, value in after.items()}
    inputs = tuple(tuple((key >> (10 - i)) & 1 for i in range(11)) for key in before)
    mapped, measurements = certified_and_channel_map(bound.circuit.instructions, inputs)
    assert measurements == 3
    assert tuple(
        sum(bit << (10 - i) for i, bit in enumerate(row)) for row in mapped
    ) == tuple(after)
    for seed in (0, 1, 19):
        assert execute_sparse_register(
            bound.circuit, before, seed=seed
        ) == pytest.approx(after, abs=1e-12)
