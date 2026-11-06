"""Source graph, phase and resource checks for indexed routing."""

from collections import Counter
from itertools import pairwise
from math import prod

import pytest

from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.swap_with_zero import (
    control_swap_with_zero_ir,
    swap_with_zero_ir,
)
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import (
    assert_complex_stream,
    assert_controlled_exchange_branches,
    assert_source_dependencies,
    source_relative_swap,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths,word_width",
    [
        ((1,), (0,), 1),
        ((3,), (2,), 2),
        ((5,), (3,), 1),
        ((2, 3), (1, 2), 2),
        ((2, 1), (1, 0), 2),
        ((1, 2), (1, 1), 1),
        ((2, 2, 2), (1, 1, 1), 1),
    ],
)
def test_source_network_phases_and_resources(
    shape: tuple[int, ...], widths: tuple[int, ...], word_width: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    selections = tuple(
        tuple(range(sum(widths[:i]), sum(widths[: i + 1]))) for i in range(len(widths))
    )
    targets = tuple(
        tuple(
            range(sum(widths) + row * word_width, sum(widths) + (row + 1) * word_width)
        )
        for row in range(prod(shape))
    )
    source = SwapWithZero(widths, word_width, shape)
    ports = dict(
        zip(
            (r.name for r in source.selection_registers),
            filter(None, selections),
            strict=True,
        )
    )
    ports["targets"] = tuple(w for word in targets for w in word)

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, CSwapApprox):
            assert all(inputs[name] == wires for name, wires in desired.items())
            return (
                source_relative_swap(inputs["ctrl"][0], inputs["x"], inputs["y"]),
                inputs,
            )
        return None

    gates = swap_with_zero_ir(selections, shape, targets)
    inverse = swap_with_zero_ir(selections, shape, targets, adjoint=True)
    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    expected_inverse, inverse_outputs = walk_source_graph(
        source.adjoint(), ports, ports, leaf
    )
    assert outputs == inverse_outputs == ports
    assert_source(gates, expected)
    assert_source(inverse, expected_inverse)
    for actual, original in [(gates, expected), (inverse, expected_inverse)]:
        assert_complex_stream(
            actual,
            original,
            inverse if actual is gates else gates,
            sum(widths) + prod(shape) * word_width,
        )
        counts = Counter(g.name for g in actual)
        assert counts["t"] + counts["tdg"] == 4 * word_width * (prod(shape) - 1)
        assert counts["cx"] == (6 * word_width - 1) * (prod(shape) - 1)
        assert counts["measure"] == 0
    cost = get_cost_value(source, QECGatesCost())
    assert cost.total_t_count() == 4 * word_width * (prod(shape) - 1)
    assert cost.measurement == 0
    assert get_cost_value(source, QubitCount()) == sum(widths) + word_width * prod(
        shape
    )


@pytest.mark.ftqc_reference
def test_original_zero_bit_axis_limitation() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    with pytest.raises(IndexError):
        SwapWithZero((0, 1), 1, (1, 2)).decompose_bloq()
    with pytest.raises(ValueError, match="earlier zero-bit"):
        swap_with_zero_ir(((), (0,)), (1, 2), ((1,), (2,)))


def test_layout_validation() -> None:
    for selections, shape, targets in [
        ((), (), ()),
        (((0,),), (3,), ((1,), (2,), (3,))),
        (((0,),), (2,), ((1,),)),
        (((0,),), (2,), ((1,), (2, 3))),
        (((0,),), (2,), ((1,), (0,))),
    ]:
        with pytest.raises(ValueError):
            swap_with_zero_ir(selections, shape, targets)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("shape", "widths", "word_width"),
    [
        ((1,), (0,), 1),
        ((3,), (2,), 2),
        ((2, 2), (1, 1), 1),
        ((2, 3), (1, 2), 1),
        ((2, 1), (1, 0), 1),
        ((1, 2), (1, 1), 1),
    ],
)
@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_outer_control_on_swap_network(
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    word_width: int,
    values: tuple[int, ...],
    adjoint: bool,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, DecomposeNotImplementedError
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    size = len(values)
    outside = tuple(range(size))
    starts = [size + sum(widths[:i]) for i in range(len(widths) + 1)]
    selections = tuple(tuple(range(a, b)) for a, b in pairwise(starts))
    targets = tuple(
        tuple(range(starts[-1] + i * word_width, starts[-1] + (i + 1) * word_width))
        for i in range(prod(shape))
    )
    data_width = starts[-1] + prod(shape) * word_width
    workspace = tuple(range(data_width, data_width + size - 1))
    bits = tuple(range(33, 33 + len(workspace)))
    args = (
        (outside,),
        (ControlRegister(1, values),),
        selections,
        shape,
        targets,
        workspace,
        bits,
    )
    actual = control_swap_with_zero_ir(*args, adjoint=adjoint)
    inverse = control_swap_with_zero_ir(*args, adjoint=not adjoint)

    base = SwapWithZero(widths, word_width, shape, uncompute=adjoint)
    source = base.controlled(CtrlSpec(cvs=values[0] if size == 1 else values))
    ports = dict(
        zip(
            (r.name for r in base.selection_registers),
            filter(None, selections),
            strict=True,
        )
    )
    ports[next(iter(source.signature)).name] = outside
    ports["targets"] = tuple(wire for word in targets for wire in word)
    expected = reference_arithmetic_gates(
        source,
        ports,
        workspace,
        dict(zip(workspace, bits, strict=True)),
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(actual, expected)
    counts = Counter(gate.name for gate in actual)
    exchanges = prod(shape) - 1
    assert counts["ccx"] == exchanges * (6 * word_width - 1)
    assert counts["ch"] == exchanges * 8 * word_width
    assert counts["cphase"] == exchanges * 12 * word_width
    assert counts["t"] + counts["tdg"] == 4 * len(workspace)
    assert counts["measure"] == len(workspace)
    assert get_cost_value(source, QubitCount()) == data_width + len(workspace)
    with pytest.raises(DecomposeNotImplementedError, match="ArbitraryClifford"):
        get_cost_value(source, QECGatesCost())
    assert_controlled_exchange_branches(
        actual, expected, inverse, data_width, workspace
    )
