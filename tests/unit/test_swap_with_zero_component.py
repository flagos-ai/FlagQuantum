"""Typed routing retains source stages and complete relative complex phases."""

from collections import Counter
from math import prod

import pytest
import torch

from flagquantum.compiler.registers.lookup.swap_with_zero import swap_with_zero_ir
from flagquantum.compiler.registers.lookup.swap_with_zero_component import (
    swap_with_zero_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "widths,shape",
    [((0,), (1,)), ((1,), (2,)), ((2,), (3,)), ((1, 1), (2, 2)), ((1, 0), (2, 1))],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_network_calls_and_full_complex_state(
    widths: tuple[int, ...], shape: tuple[int, ...], adjoint: bool
) -> None:
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    word = 1
    program = swap_with_zero_program(widths, word, shape, adjoint=adjoint)
    source = SwapWithZero(widths, word, shape, uncompute=adjoint)
    signature = Signature(program.functions[0].registers or ())
    assert _native_trace(program) == _source_trace(
        _source(signature), source.decompose_bloq()
    )
    selections = tuple(
        tuple(range(sum(widths[:i]), sum(widths[: i + 1]))) for i in range(len(widths))
    )
    targets = tuple((sum(widths) + i,) for i in range(prod(shape)))
    bound = bind_register_program(
        program, (*(selection for selection in selections if selection), *targets)
    )
    previous = swap_with_zero_ir(selections, shape, targets, adjoint=adjoint)
    assert bound.events == ()
    assert bound.outputs == (
        *(selection for selection in selections if selection),
        *targets,
    )
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in previous
    )
    dimension = 1 << (sum(widths) + prod(shape))
    index = torch.arange(dimension, dtype=torch.float64)
    state = torch.complex(index + 1, (index % 7 - 3) * index)
    state /= state.norm()
    torch.testing.assert_close(
        forced_branch(state, bound.circuit.instructions, ()),
        forced_branch(state, previous, ()),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "widths,word,shape",
    [
        ((-1,), 1, (1,)),
        ((1,), 0, (2,)),
        ((1,), 1, (3,)),
        ((0, 1), 1, (1, 2)),
        ((1,), 1, ()),
    ],
)
def test_network_admission_reuses_existing_refusals(
    widths: tuple[int, ...], word: int, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        swap_with_zero_program(widths, word, shape)
