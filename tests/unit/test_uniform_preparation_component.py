"""Stored uniform recipes retain complete source call graphs and complex action."""

import pytest
import torch

from flagquantum.compiler.registers.preparation.uniform_preparation import (
    uniform_preparation_requirements,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_component import (
    uniform_preparation_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 10, 12])
@pytest.mark.parametrize("values", [(), (1,), (0,), (1, 0)])
def test_uniform_full_source_calls_and_complete_small_operators(
    count: int, values: tuple[int, ...], adjoint: bool
) -> None:
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    source = PrepareUniformSuperposition(count, cvs=values)
    if adjoint:
        source = source.adjoint()
    graph = uniform_preparation_program(count, values, adjoint=adjoint)
    compare_layers(graph, source, atomic_leaves(graph))
    assert verify_program_record(graph) == ()
    width, work, _ = uniform_preparation_requirements(count, len(values))
    work = max(work, len(values) - 1)
    logical = width + len(values)
    inputs = (
        *((tuple(range(len(values))),) if values else ()),
        tuple(range(len(values), logical)),
    )
    bound = bind_register_program(
        graph, inputs, clean=tuple(range(logical, logical + work))
    )
    if count <= 5 and len(values) <= 1:
        expected = torch.as_tensor(source.tensor_contract())
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical), expected, atol=1e-11, rtol=0
        )


@pytest.mark.parametrize("values", [(), (0,), (1,), (1, 0)])
def test_odd_uniform_all_small_histories_and_exact_recovery(
    values: tuple[int, ...],
) -> None:
    from itertools import product

    from tests.unit.measurement_checks import forced_branch

    count = 3
    width, work, _ = uniform_preparation_requirements(count, len(values))
    work = max(work, len(values) - 1)
    logical = width + len(values)
    inputs = (
        *((tuple(range(len(values))),) if values else ()),
        tuple(range(len(values), logical)),
    )
    clean = tuple(range(logical, logical + work))
    forward = bind_register_program(
        uniform_preparation_program(count, values), inputs, clean=clean
    ).circuit
    backward = bind_register_program(
        uniform_preparation_program(count, values, adjoint=True), inputs, clean=clean
    ).circuit
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    expected = torch.as_tensor(
        PrepareUniformSuperposition(count, cvs=values).tensor_contract(),
        dtype=torch.complex128,
    )
    stride = 1 << work
    state = torch.zeros(1 << forward.n_wires, dtype=torch.complex128)
    state[::stride] = torch.tensor([complex(i + 1, 2 - i) for i in range(1 << logical)])
    state /= state.norm()
    target = torch.zeros_like(state)
    target[::stride] = expected @ state[::stride]
    for stream, initial, desired in [
        (forward.instructions, state, target),
        (backward.instructions, target, state),
    ]:
        events = sum(g.name == "measure" for g in stream)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(initial, stream, history),
                desired / 2 ** (events / 2),
                atol=2e-11,
                rtol=0,
            )


def test_symbolic_declaration_and_original_singleton_refusal() -> None:
    from sympy import Symbol

    from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls

    record = uniform_preparation_program(Symbol("n", positive=True, integer=True))
    assert verify_program_record(record) == ()
    assert record.functions[0].blocks == ()
    with pytest.raises(ValueError, match="singleton"):
        uniform_preparation_program(1)
    with pytest.raises(ValueError, match="binary"):
        uniform_preparation_program(3, (2,))
    with pytest.raises(ValueError, match="Missing exact component adjoint"):
        adjoint_component_calls(uniform_preparation_program(2), ())


@pytest.mark.parametrize("width", [1, 2, 3])
def test_constant_comparison_adjoint_retains_source_calls_for_every_threshold(
    width: int,
) -> None:
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant

    from flagquantum.compiler.registers.comparison.constant_component import (
        less_than_constant_program,
    )

    for threshold in range((1 << width) + 1):
        source = LessThanConstant(width, threshold).adjoint()
        graph = less_than_constant_program(width, threshold, adjoint=True)
        compare_layers(graph, source, atomic_leaves(graph))
        assert verify_program_record(graph) == ()
