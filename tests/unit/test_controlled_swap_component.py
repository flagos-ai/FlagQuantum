"""Exact controlled exchange retains opaque word calls and coherent phases."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.primitives.controlled_swap_component import (
    controlled_swap_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
def test_exchange_source_calls_stream_and_separate_cost_conventions(width: int) -> None:
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    graph, source = controlled_swap_program(width), CSwap(width)
    compare_layers(graph, source, atomic_leaves(graph))
    ports = {
        "ctrl": (0,),
        "x": tuple(range(1, width + 1)),
        "y": tuple(range(width + 1, 2 * width + 1)),
    }
    bound = bind_register_program(graph, tuple(ports.values()))
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(source, ports, (), {}, preserve_components=True),
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert (
        counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7) == 7 * width
    )
    assert model.total_t_count() == 4 * width
    assert counts["measure"] == 0


@pytest.mark.parametrize("width", (1, 2, 3))
def test_exchange_full_coherent_source_tensor_and_self_inverse(width: int) -> None:
    from qualtran.bloqs.basic_gates import CSwap

    logical = 2 * width + 1
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    initial = torch.complex(raw, 7 - 3 * raw)
    initial /= initial.norm()
    bound = bind_register_program(
        controlled_swap_program(width),
        ((0,), tuple(range(1, width + 1)), tuple(range(width + 1, 2 * width + 1))),
    )
    matrix = torch.as_tensor(
        CSwap(width).tensor_contract(), dtype=torch.complex128
    ).reshape(1 << logical, 1 << logical)
    expected = (matrix @ initial.reshape(-1, 2)).flatten()
    actual = forced_branch(initial, bound.circuit.instructions, ())
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    torch.testing.assert_close(
        forced_branch(actual, bound.circuit.instructions, ()),
        initial,
        rtol=0,
        atol=3e-12,
    )
