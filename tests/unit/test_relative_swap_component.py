"""Relative-swap typed calls preserve original imports and complex operators."""

from collections import Counter
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
    relative_controlled_swap_ir,
)
from flagquantum.compiler.registers.primitives.relative_swap_component import (
    relative_swap_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_original_child_order_and_complex_operator(width: int) -> None:
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = CSwapApprox(width)
    program = relative_swap_program(width)
    assert verify_program_record(program) == ()
    signature = Signature(program.functions[0].registers or ())
    assert _native_trace(program) == _source_trace(
        _source(signature), original.decompose_bloq()
    )
    bound = bind_register_program(
        program,
        ((0,), tuple(range(1, 1 + width)), tuple(range(1 + width, 1 + 2 * width))),
    )
    physical = relative_controlled_swap_ir(
        0, tuple(range(1, 1 + width)), tuple(range(1 + width, 1 + 2 * width))
    )
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in physical
    )
    assert sum(g.name in {"t", "tdg"} for g in bound.circuit.instructions) == 4 * width
    assert get_cost_value(original, QECGatesCost()).t == 4 * width
    assert bound.events == ()
    dimension = 1 << (1 + 2 * width)
    amplitudes = torch.arange(dimension, dtype=torch.float64)
    state = torch.complex(amplitudes + 1, (amplitudes % 7 - 3) * amplitudes)
    state /= state.norm()
    actual = forced_branch(state, bound.circuit.instructions, ())
    expected = forced_branch(state, physical, ())
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    if width <= 2:
        matrix = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
        torch.testing.assert_close(actual, matrix @ state, atol=1e-12, rtol=0)


@pytest.mark.parametrize("width", (0, -1, True, 1.5))
def test_relative_swap_program_width_refusals(width: Any) -> None:
    with pytest.raises(ValueError, match="positive concrete integer"):
        relative_swap_program(width)
