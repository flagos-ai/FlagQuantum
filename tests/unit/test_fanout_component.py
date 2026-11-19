"""Shared typed fanout preserves the source tree on arbitrary dirty targets."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.bits.fanout_component import (
    fanout_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8, 16))
def test_source_gate_dependencies_without_work(width: int) -> None:
    from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    inputs = ((width,), tuple(reversed(range(width))))
    bound = bind_register_program(fanout_component_program(width), inputs)
    source = MultiTargetCNOT(width)
    original = reference_arithmetic_gates(
        source,
        dict(zip(("control", "targets"), inputs, strict=True)),
        (),
        {},
        cirq_style=True,
    )
    assert_source(bound.circuit.instructions, original)
    assert bound.outputs == inputs and bound.events == ()
    assert Counter(g.name for g in bound.circuit.instructions) == {"cx": 2 * width - 1}
    # Source QEC treats the entire Clifford fanout as one logical operation.
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == 1 and model.total_t_count() == model.measurement == 0
    assert get_cost_value(source, QubitCount()) == width + 1


@pytest.mark.parametrize("width", (1, 2, 3, 5))
def test_source_full_complex_tensor_dirty_targets_and_double_append(width: int) -> None:
    from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT

    source = MultiTargetCNOT(width)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << (width + 1), -1
    )
    raw = torch.arange(1, (1 << (width + 2)) + 1, dtype=torch.float64)
    state = torch.complex(raw, 3 - 5 * raw)
    state /= state.norm()
    bound = bind_register_program(
        fanout_component_program(width), ((0,), tuple(range(1, width + 1)))
    )
    result = forced_branch(state, bound.circuit.instructions, ())
    expected = (matrix @ state.reshape(-1, 2)).flatten()
    torch.testing.assert_close(result, expected, rtol=0, atol=2e-12)
    torch.testing.assert_close(
        forced_branch(result, bound.circuit.instructions, ()), state, rtol=0, atol=2e-12
    )
