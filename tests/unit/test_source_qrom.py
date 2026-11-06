"""Concrete QROM source gates and complete measured lookup channels."""

from dataclasses import asdict
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.qrom_resource_model import source_qrom_cost
from flagquantum.compiler.registers.source_qrom import (
    source_qrom_ir,
    source_qrom_workspace_size,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def source_qrom_gates(
    source: Any,
    ports: dict[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    return reference_arithmetic_gates(
        source,
        ports,
        workspace,
        dict(zip(workspace, bits, strict=True)),
        cirq_style=True,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "controls_count,shape,widths,target_layout,datasets",
    [
        (0, (3,), (3,), ((1, 2),), ((1, 2, 3),)),
        (1, (3,), (2,), ((1, 2),), ((1, 2, 3),)),
        (0, (4,), (2,), ((1, 2),), ((1, 1, 0, 2),)),
        (2, (4,), (2,), ((1, 1),), ((1, 1, 0, 1),)),
        (3, (2,), (1,), ((1, 1),), ((1, 0),)),
        (0, (2, 3), (1, 2), ((1, 2),), ((0, 1, 2, 3, 0, 3),)),
        (1, (2, 2), (1, 1), ((2, 1), (1, 1)), ((1, 0, 1, 0, 1, 0, 0, 1), (1, 1, 0, 1))),
        (2, (1,), (0,), ((2, 1),), ((1, 0),)),
        (0, (1, 2), (0, 1), ((1, 1),), ((1, 1),)),
        (0, (4,), (2,), ((1, 1),), ((0, 0, 0, 0),)),
        (0, (), (), ((1, 2),), ((3,),)),
    ],
)
def test_literal_qrom_and_complete_channel(
    controls_count: int,
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    target_layout: tuple[tuple[int, int], ...],
    datasets: tuple[tuple[int, ...], ...],
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    cursor = 0

    def alloc(count: int) -> tuple[int, ...]:
        nonlocal cursor
        start = cursor
        cursor += count
        return tuple(range(start, cursor))

    controls = alloc(controls_count)
    selections = tuple(alloc(width) for width in widths)
    targets = tuple(
        tuple(alloc(width) for _ in range(count)) for count, width in target_layout
    )
    data_width = sum(count * width for count, width in target_layout)
    work_size = source_qrom_workspace_size(controls_count, widths, datasets)
    workspace = alloc(work_size)
    bits = tuple(range(40, 40 + work_size))
    target_shapes = tuple(() if count == 1 else (count,) for count, _ in target_layout)
    source = QROM(
        tuple(
            np.array(data).reshape((*shape, *target_shape))
            for data, target_shape in zip(datasets, target_shapes, strict=True)
        ),
        widths,
        tuple(width for _, width in target_layout),
        target_shapes,
        num_controls=controls_count,
    )
    ports = {"control": controls} if controls else {}
    ports.update(
        zip(
            (reg.name for reg in source.selection_registers),
            filter(None, selections),
            strict=True,
        )
    )
    ports.update(
        (f"target{i}_", tuple(w for word in group for w in word))
        for i, group in enumerate(targets)
        if group[0]
    )
    gates = source_qrom_ir(
        controls, selections, shape, targets, datasets, workspace, bits
    )
    original = source_qrom_gates(source, ports, workspace, bits)
    assert_source(gates, original)
    events = sum(g.name == "measure" for g in gates)
    assert sum(g.name in {"t", "tdg"} for g in gates) == 4 * events
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == 4 * events
    assert model.measurement == events
    estimated = estimate_component_resources(
        source_qrom_cost(controls_count, widths, shape, target_layout, datasets)
    )
    assert {
        key: value for key, value in asdict(estimated.gates).items() if value
    } == model.asdict()
    assert estimated.peak_logical_qubits == get_cost_value(source, QubitCount())
    # The static estimate always includes the full selection pool.
    assert (
        get_cost_value(source, QubitCount())
        == cursor - work_size + sum(widths) + controls_count - 1
    )

    inputs, outputs = [], []
    for control, addr, initial_target in product(
        range(1 << controls_count),
        product(*(range(extent) for extent in shape)),
        range(1 << data_width),
    ):
        row = sum(index * prod(shape[axis + 1 :]) for axis, index in enumerate(addr))
        address = control
        for index, width in zip(addr, widths, strict=True):
            address = (address << width) | index
        mask = 0
        if control == (1 << controls_count) - 1:
            for (count, width), data in zip(target_layout, datasets, strict=True):
                for word in data[row * count : (row + 1) * count]:
                    mask = (mask << width) | word
        inputs.append(((address << data_width) | initial_target) << work_size)
        outputs.append(((address << data_width) | (initial_target ^ mask)) << work_size)

    def columns(states: list[int]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple((i >> bit) & 1 for bit in reversed(range(cursor))) for i in states
        )

    assert certified_and_channel_map(gates, columns(inputs)) == (
        columns(outputs),
        events,
    )

    # Exhaust every raw history for small circuits, including a coherent spectator.
    if events <= 5 and cursor <= 11:
        state = torch.zeros(1 << (cursor + 1), dtype=torch.complex128)
        expected = torch.zeros_like(state)
        weights = torch.arange(1, 2 * len(inputs) + 1, dtype=torch.float64)
        coefficients = torch.complex(weights, weights * (weights % 5 - 2))
        coefficients /= coefficients.norm()
        state[[2 * i + spectator for i in inputs for spectator in (0, 1)]] = (
            coefficients
        )
        expected[[2 * i + spectator for i in outputs for spectator in (0, 1)]] = (
            coefficients
        )
        for outcomes in product((0, 1), repeat=events):
            actual = forced_branch(state, gates, outcomes)
            torch.testing.assert_close(
                actual, expected * 2 ** (-events / 2), atol=2e-12, rtol=0
            )


@pytest.mark.ftqc_reference
def test_locked_source_singleton_failure() -> None:
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.data_loading.qrom import QROM

    source = QROM((np.array([[0, 1]]),), (0, 1), (1,), ((),))
    with pytest.raises(TypeError, match="0-dimensional"):
        tuple(
            cirq.flatten_op_tree(
                cirq.decompose_once(source.on(*cirq.LineQubit.range(2)))
            )
        )
    with pytest.raises(ValueError, match="mixed singleton"):
        source_qrom_ir((), ((), (0,)), (1, 2), (((1,),),), ((0, 1),), (), ())


@pytest.mark.parametrize(
    "selection,target,data,work,bits",
    [
        ((0,), (1,), (0, 1, 0), (), ()),
        ((0, 1), (2,), (0, 1, 2), (3,), (40,)),
        ((0, 1), (2,), (0, 1, 0), (), ()),
        ((0, 1), (2,), (0, 1, 0), (3,), (-1,)),
        ((0, 1), (1,), (0, 1, 0), (3,), (40,)),
        ((0, 1), (), (0, 0, 0), (), ()),
    ],
)
def test_rejects_invalid_qrom_bindings(
    selection: tuple[int, ...],
    target: tuple[int, ...],
    data: tuple[int, ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        source_qrom_ir((), (selection,), (3,), ((target,),), (data,), work, bits)


def test_rejects_missing_target_group() -> None:
    with pytest.raises(ValueError, match="one target group per dataset"):
        source_qrom_ir((), (), (), (), ((1,),), (), ())
