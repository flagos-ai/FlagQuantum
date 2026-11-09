"""Verify source control wrappers and their complete complex trajectories."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler._component_trace.compound_control_resource import (
    compound_control_key,
    controlled_swap_cost_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledSwapCall,
    FixedTCall,
    MergedFredkinCall,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.controlled_swap import (
    control_bit_cswap_ir,
    control_register_cswap_ir,
    controlled_swap_with_predicate_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
def test_controlled_swap_source_gates_models_and_all_branches(
    width: int, values: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.swap import Swap, TwoBitCSwap, TwoBitSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    size = len(values)
    controls = tuple(range(size))
    left = tuple(range(size, size + width))
    right = tuple(range(size + width, size + 2 * width))
    workspace = tuple(range(size + 2 * width, 2 * size + 2 * width - 1))
    bits = tuple(range(25, 25 + len(workspace)))
    gates = controlled_swap_with_predicate_ir(
        (controls,), (ControlRegister(1, values),), left, right, workspace, bits
    )
    spec = CtrlSpec(cvs=values[0] if size == 1 else values)
    source = Swap(width).controlled(spec)
    sources = [(source, controlled_swap_cost_key(width, values))]
    if width == 1:
        key = compound_control_key(CliffordCall("swap_bit", 2), values)
        assert key is not None
        sources.append((TwoBitSwap().controlled(spec), key))
    counts = Counter(gate.name for gate in gates)
    for upstream, key in sources:
        reference = reference_arithmetic_gates(
            (
                upstream.as_composite_bloq()
                if isinstance(upstream, TwoBitCSwap)
                else upstream
            ),
            {"ctrl": controls, "x": left, "y": right},
            workspace,
            dict(zip(workspace, bits, strict=True)),
            preserve_components=True,
        )
        assert_source(gates, reference)
        estimate = get_cost_value(upstream, QECGatesCost())
        modeled = query_qec_gate_cost(key)
        assert {
            name: value for name, value in asdict(modeled).items() if value
        } == estimate.asdict()
        assert counts["t"] + counts["tdg"] == estimate.total_t_count(ts_per_cswap=7)
        assert counts["measure"] == counts["reset"] == estimate.measurement == size - 1
        assert get_cost_value(upstream, QubitCount()) == size + 2 * width + len(
            workspace
        )

    assert_swap_branches(gates, width, values, workspace, size - 1)


def assert_swap_branches(
    gates: tuple[Instruction, ...],
    width: int,
    values: tuple[int, ...],
    workspace: tuple[int, ...],
    measurements: int,
) -> None:
    logical = len(values) + 2 * width
    wires = logical + len(workspace) + 1
    data = torch.arange(1 << (logical + 1))
    spectator = data & 1
    operands = data >> 1
    control = operands >> (2 * width)
    mask = (1 << width) - 1
    x, y = (operands >> width) & mask, operands & mask
    activation = int("".join(map(str, values)), 2)
    swapped = (control << (2 * width)) | (y << width) | x
    output = torch.where(control == activation, swapped, operands)
    state = torch.zeros(1 << wires, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    amplitudes = torch.complex(data.double() + 1, (data % 7).double() - 3)
    amplitudes /= amplitudes.norm()
    state[(operands << (len(workspace) + 1)) | spectator] = amplitudes
    expected[(output << (len(workspace) + 1)) | spectator] = amplitudes
    scale = sqrt(2**measurements)
    for outcomes in product((0, 1), repeat=measurements):
        result = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(result, expected / scale, atol=2e-12, rtol=0)
        torch.testing.assert_close(
            forced_branch(result, gates, outcomes), state / scale**2, atol=2e-12, rtol=0
        )


def test_swap_controls_reject_aliases() -> None:
    with pytest.raises(ValueError, match="distinct"):
        controlled_swap_with_predicate_ir(
            ((0, 1),), (ControlRegister(1, (0, 1)),), (2,), (3,), (3,), (0,)
        )


@pytest.mark.ftqc_reference
def test_symbolic_predicate_swap_resource_formula() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.swap import Swap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol

    n = Symbol("n", integer=True, positive=True)
    values = (0, 1)
    source = Swap(n).controlled(CtrlSpec(cvs=values))
    cost = query_qec_gate_cost(controlled_swap_cost_key(n, values))
    assert {key: value for key, value in asdict(cost).items() if value != 0} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == 2 * n + 3


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
def test_outer_control_on_register_cswap(width: int, values: tuple[int, ...]) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    size = len(values)
    controls = tuple(range(size))
    original_control = size
    left = tuple(range(size + 1, size + width + 1))
    right = tuple(range(size + width + 1, size + 2 * width + 1))
    workspace = tuple(range(size + 2 * width + 1, 2 * size + 2 * width + 1))
    bits = tuple(range(30, 30 + size))
    gates = control_register_cswap_ir(
        (controls,),
        (ControlRegister(1, values),),
        original_control,
        left,
        right,
        workspace,
        bits,
    )
    source = CSwap(width).controlled(CtrlSpec(cvs=values[0] if size == 1 else values))
    # Use the pinned signature to bind the inherited and added controls.
    names = [register.name for register in source.signature]
    inner_name, outer_name = names[1], names[0]
    reference = reference_arithmetic_gates(
        source,
        {outer_name: controls, inner_name: (original_control,), "x": left, "y": right},
        workspace,
        dict(zip(workspace, bits, strict=True)),
        preserve_components=True,
    )
    assert_source(gates, reference)

    key = compound_control_key(ControlledSwapCall(width), values)
    assert key is not None
    modeled = query_qec_gate_cost(key)
    cost = get_cost_value(source, QECGatesCost())
    assert {name: n for name, n in asdict(modeled).items() if n} == cost.asdict()
    counts = Counter(gate.name for gate in gates)
    assert counts["t"] + counts["tdg"] == cost.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == cost.measurement == width + size - 1
    assert get_cost_value(source, QubitCount()) == size + 1 + 2 * width + len(workspace)
    assert_swap_branches(gates, width, (*values, 1), workspace, cost.measurement)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("prior", "outer"),
    [
        ((), (0,)),
        ((), (1,)),
        ((), (0, 1)),
        ((), (1, 0, 1)),
        ((0,), (0,)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((1,), (1,)),
        ((0, 1), (0,)),
        ((1, 0, 0), (1,)),
    ],
)
def test_atomic_fredkin_control_specialization(
    prior: tuple[int, ...], outer: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = (*reversed(prior), 1)
    size = len(outer) + len(original)
    outside = tuple(range(len(outer)))
    inside = tuple(range(len(outer), size))
    left, right = size, size + 1
    workspace = tuple(range(size + 2, 2 * size + 1))
    bits = tuple(range(30, 30 + len(workspace)))
    gates = control_bit_cswap_ir(
        (outside,),
        (ControlRegister(1, outer),),
        inside,
        original,
        left,
        right,
        workspace,
        bits,
    )
    child = TwoBitCSwap()
    for value in prior:
        child = child.controlled(CtrlSpec(cvs=value))
    source = child.controlled(CtrlSpec(cvs=outer[0] if len(outer) == 1 else outer))
    names = [register.name for register in source.signature]
    bindings = (
        {names[0]: (*outside, *inside)}
        if len(outer) == 1
        else {names[0]: outside, names[1]: inside}
    )
    reference = reference_arithmetic_gates(
        source,
        {**bindings, "x": (left,), "y": (right,)},
        workspace,
        dict(zip(workspace, bits, strict=True)),
        preserve_components=True,
    )
    assert_source(gates, reference)

    key = compound_control_key(
        MergedFredkinCall(original) if prior else FixedTCall("fredkin"), outer
    )
    assert key is not None
    cost = get_cost_value(source, QECGatesCost())
    assert {
        name: count for name, count in asdict(query_qec_gate_cost(key)).items() if count
    } == cost.asdict()
    counts = Counter(gate.name for gate in gates)
    assert counts["t"] + counts["tdg"] == cost.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == cost.measurement == size - 1
    assert get_cost_value(source, QubitCount()) == size + 2 + len(workspace)
    assert_swap_branches(gates, 1, (*outer, *original), workspace, cost.measurement)


@pytest.mark.ftqc_reference
def test_atomic_fredkin_rejects_pinned_undecomposable_control() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.swap import TwoBitCSwap

    source = TwoBitCSwap().controlled(CtrlSpec(cvs=0)).controlled(CtrlSpec(cvs=(0, 1)))
    with pytest.raises(TypeError, match="only integer scalar arrays"):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="no pinned decomposition"):
        control_bit_cswap_ir(
            ((0, 1),),
            (ControlRegister(1, (0, 1)),),
            (2, 3),
            (0, 1),
            4,
            5,
            (6, 7, 8),
            (30, 31, 32),
        )
