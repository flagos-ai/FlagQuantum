"""Source empty sorting networks preserve zero storage and spectator phases."""

from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_sorting_factories import sorting_factories
from flagquantum.compiler.registers.comparison.parallel_comparison import (
    parallel_compare_ir,
)
from flagquantum.compiler.registers.comparison.parallel_comparison_component import (
    parallel_comparison_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.bitonic_sort import bitonic_sort_ir
from flagquantum.compiler.registers.sorting.bitonic_sort_component import (
    bitonic_sort_program,
)
from flagquantum.core.ir.program import verify_program_record
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,count,offset,width",
    (
        ("parallel", 0, 1, 2),
        ("parallel", 0, 7, 0),
        ("parallel", 1, 2, 0),
        ("parallel", 3, 5, 0),
        ("parallel", 3, 5, 2),
        ("sort", 1, 1, 0),
        ("sort", 1, 1, 2),
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_empty_graph_sided_ports_zero_resources_and_complex_spectator(
    kind: str, count: int, offset: int, width: int, adjoint: bool
) -> None:
    from qualtran import BloqError
    from qualtran.bloqs.arithmetic.sorting import BitonicSort, ParallelComparators
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = (
        ParallelComparators(count, offset, width)
        if kind == "parallel"
        else BitonicSort(count, width)
    )
    owner = (
        parallel_comparison_program(count, offset, width, adjoint=adjoint)
        if kind == "parallel"
        else bitonic_sort_program(count, width, adjoint=adjoint)
    )
    if adjoint:
        source = source.adjoint()
    if adjoint and kind == "parallel" and count == 0:
        with pytest.raises(BloqError, match="Finalizing requires a Soquet named"):
            source.decompose_bloq()
    else:
        assert not source.decompose_bloq().bloq_instances
    assert verify_program_record(owner) == ()
    assert owner.functions[0].registers is not None
    history = owner.functions[0].registers[-1]
    assert history.shape_symbolic == (0,) and history.side.name == (
        "LEFT" if adjoint else "RIGHT"
    )
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(count))
    spectator = count * width
    bound = bind_register_program(owner, words, borrowed=(spectator,))
    assert bound.circuit.instructions == () and bound.outputs == words
    assert not bound.events
    if width == 0:
        with pytest.raises(BloqError, match="GreaterThan does not accept Soquets"):
            get_cost_value(source, QECGatesCost())
    else:
        model = get_cost_value(source, QECGatesCost())
        assert model.total_t_count(ts_per_cswap=7) == model.measurement == 0
    size = spectator + 1
    raw = torch.arange(1, (1 << size) + 1, dtype=torch.float64)
    state = torch.complex(raw, 5 - 2 * raw)
    state /= state.norm()
    assert torch.equal(forced_branch(state, bound.circuit.instructions, ()), state)
    actual = (
        Circuit(size, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(size, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, state, atol=1e-15, rtol=0)
    if not adjoint:
        program = resolve_component_program(_library(source, 0), 0, sorting_factories())
        assert bind_register_program(program, words, borrowed=(spectator,)) == bound
    primitive = (
        parallel_compare_ir(words, offset, (), (), (), adjoint=adjoint)
        if kind == "parallel"
        else bitonic_sort_ir(words, (), (), (), adjoint=adjoint)
    )
    assert primitive == ()


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize(
    "histories,workspace,slots", (((1,), (), ()), ((), (1,), ()), ((), (), (1,)))
)
def test_empty_layout_rejects_spurious_storage(
    adjoint: bool,
    histories: tuple[int, ...],
    workspace: tuple[int, ...],
    slots: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="empty"):
        parallel_compare_ir((), 1, histories, workspace, slots, adjoint=adjoint)


@pytest.mark.parametrize("width", (-1, True, 1.0))
def test_empty_component_keeps_width_validation(width: Any) -> None:
    with pytest.raises(ValueError, match="width"):
        parallel_comparison_program(0, 1, width)
    with pytest.raises(ValueError, match="width"):
        bitonic_sort_program(1, width)
