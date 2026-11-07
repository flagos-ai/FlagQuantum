"""Exercise untouched composite adjoints, branch phases and indexed round trips."""

from collections import Counter
from itertools import product
from math import pi, prod
from typing import Literal

import pytest
import torch

from flagquantum.compiler.registers.indexed_component import (
    indexed_component_ir,
    indexed_pauli_targets_ir,
)
from flagquantum.compiler.registers.phase_family import phase_family_ir
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_indexed_component import _layout
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit
Pauli = Literal["i", "x", "y", "z"]
CASES = (
    ((2,), (1,), 0),
    ((2,), (1,), 1),
    ((2,), (1,), 2),
    ((2,), (1,), 3),
    ((3,), (2,), 0),
    ((3,), (2,), 1),
    ((3,), (4,), 1),
    ((2, 2), (1, 1), 0),
    ((2, 2), (1, 1), 1),
    ((2, 3), (1, 2), 1),
    ((2, 2, 2), (1, 1, 1), 0),
    ((2, 2, 2), (1, 1, 1), 1),
    ((1, 2), (1, 1), 1),
)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths,controls,position",
    [
        (*case, position)
        for case in CASES
        for position in (False, True)
        if position or case[2] <= 1
    ],
)
@pytest.mark.parametrize("pattern", ["xz", "iy"])
def test_original_composite_adjoint_stream_and_resources(
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    controls: int,
    position: bool,
    pattern: str,
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import BQUInt, QAny, Register
    from qualtran.bloqs.basic_gates import Identity, XGate, YGate, ZGate
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    ctrl, axes, targets, work, bits = _layout(
        widths, controls, prod(shape) if position else 1
    )
    labels: tuple[Pauli, ...] = tuple(
        pattern[index % len(pattern)] for index in range(prod(shape))
    )
    regs = tuple(
        Register(f"s{index}", BQUInt(width, size))
        for index, (width, size) in enumerate(zip(widths, shape, strict=True))
    )
    ports = {reg.name: word for reg, word in zip(regs, axes, strict=True)}
    if controls:
        ports["control"] = ctrl
    if position:
        source = ApplyGateToLthQubit(
            regs,
            lambda *indices: {"i": cirq.I, "x": cirq.X, "y": cirq.Y, "z": cirq.Z}[
                labels[int(np.ravel_multi_index(indices, shape))]
            ],
            control_regs=(Register("control", QAny(controls)),) if controls else (),
        )
        ports["target"] = targets
        actual = indexed_pauli_targets_ir(
            axes,
            shape,
            targets,
            labels,
            controls=ctrl,
            workspace=work,
            classical_bits=bits,
            adjoint=True,
        )
    else:
        source = ApplyLthBloq(
            np.array(
                [
                    {"i": Identity, "x": XGate, "y": YGate, "z": ZGate}[label]()
                    for label in labels
                ],
                dtype=object,
            ).reshape(shape),
            selection_regs=regs,
            control_val=1 if controls else None,
        )
        ports["q"] = targets
        actual = indexed_component_ir(
            axes,
            shape,
            lambda index, predicate: (
                ()
                if labels[index] == "i"
                else (Instruction("c" + labels[index], (predicate, targets[0])),)
            ),
            controls=ctrl,
            workspace=work,
            classical_bits=bits,
            adjoint=True,
        )
    original_adjoint = source.adjoint()
    composite = original_adjoint.decompose_bloq()
    # Composite export omits unused reserved workspace. Bind its actual storage
    # to the same physical parent wires, preserving the caller's unused slots.
    occupied = tuple(
        wire for wire in work if any(wire in gate.wires for gate in actual)
    )
    expected = reference_arithmetic_gates(
        composite,
        ports,
        occupied,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(actual, expected)
    counts = Counter(gate.name for gate in actual)
    model = get_cost_value(original_adjoint, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement
    assert get_cost_value(original_adjoint, QubitCount()) == len(ctrl) + sum(
        widths
    ) + len(targets) + len(occupied)


@pytest.mark.parametrize(
    "shape,widths,controlled",
    [
        ((2,), (1,), False),
        ((2,), (1,), True),
        ((3,), (2,), True),
        ((2, 2), (1, 1), False),
        ((2, 2), (1, 1), True),
    ],
)
def test_nonselfadjoint_children_all_complex_branches_and_round_trip(
    shape: tuple[int, ...], widths: tuple[int, ...], controlled: bool
) -> None:
    ctrl, axes, targets, work, bits = _layout(widths, int(controlled), 1)
    data = len(ctrl) + sum(widths) + 1
    total = data + len(work)
    directions = []
    for adjoint in (False, True):
        directions.append(
            indexed_component_ir(
                axes,
                shape,
                lambda index, predicate, direction=adjoint: phase_family_ir(
                    "s" if index % 2 == 0 else "t",
                    targets[0],
                    controls=(predicate,),
                    values=(1,),
                    adjoint=direction,
                ),
                controls=ctrl,
                workspace=work,
                classical_bits=bits,
                adjoint=adjoint,
            )
        )
    generator = torch.Generator().manual_seed(47)
    amplitudes = torch.complex(
        torch.randn(1 << data, generator=generator, dtype=torch.float64),
        torch.randn(1 << data, generator=generator, dtype=torch.float64),
    )
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    inverse_expected = torch.zeros_like(initial)
    for word in range(1 << data):
        indices = [
            sum(
                ((word >> (data - 1 - wire)) & 1) << (len(axis) - 1 - index)
                for index, wire in enumerate(axis)
            )
            for axis in axes
        ]
        if any(index >= extent for index, extent in zip(indices, shape, strict=True)):
            continue
        flat = 0
        for index, extent in zip(indices, shape, strict=True):
            flat = flat * extent + index
        initial[word << len(work)] = amplitudes[word]
        active = all((word >> (data - 1 - wire)) & 1 for wire in ctrl)
        angle = -(pi / 2 if flat % 2 == 0 else pi / 4) if active and word & 1 else 0
        inverse_expected[word << len(work)] = amplitudes[word] * complex(
            torch.exp(torch.tensor(1j * angle, dtype=torch.complex128))
        )
    norm = initial.norm()
    initial /= norm
    inverse_expected /= norm
    back_events = sum(gate.name == "measure" for gate in directions[1])
    for outcomes in product((0, 1), repeat=back_events):
        branch = forced_branch(initial, directions[1], outcomes)
        torch.testing.assert_close(
            branch, inverse_expected / (2 ** (back_events / 2)), atol=2e-13, rtol=2e-13
        )
    pair = (*directions[0], *directions[1])
    events = sum(gate.name == "measure" for gate in pair)
    for outcomes in product((0, 1), repeat=events):
        branch = forced_branch(initial, pair, outcomes)
        torch.testing.assert_close(
            branch, initial / (2 ** (events / 2)), atol=2e-13, rtol=2e-13
        )


def test_direction_is_validated_before_child_callback() -> None:
    def unexpected(index: int, predicate: int) -> tuple[Instruction, ...]:
        pytest.fail("Invalid traversal direction cannot reach a child")

    with pytest.raises(ValueError, match="direction"):
        indexed_component_ir(
            ((0,),), (2,), unexpected, workspace=(), classical_bits=(), adjoint=1
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths,controlled",
    [((2,), (1,), False), ((2,), (1,), True), ((2, 2), (1, 1), True)],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_nonselfadjoint_phase_child_stream(
    shape: tuple[int, ...], widths: tuple[int, ...], controlled: bool, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran import BQUInt, Register
    from qualtran.bloqs.basic_gates import SGate, TGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    ctrl, axes, targets, work, bits = _layout(widths, int(controlled), 1)
    regs = tuple(
        Register(f"s{index}", BQUInt(width, size))
        for index, (width, size) in enumerate(zip(widths, shape, strict=True))
    )
    source = ApplyLthBloq(
        np.array(
            [SGate() if index % 2 == 0 else TGate() for index in range(prod(shape))],
            dtype=object,
        ).reshape(shape),
        selection_regs=regs,
        control_val=1 if controlled else None,
    )
    ports = {reg.name: word for reg, word in zip(regs, axes, strict=True)}
    ports["q"] = targets
    if controlled:
        ports["control"] = ctrl
    actual = indexed_component_ir(
        axes,
        shape,
        lambda index, predicate: phase_family_ir(
            "s" if index % 2 == 0 else "t",
            targets[0],
            controls=(predicate,),
            values=(1,),
            adjoint=adjoint,
        ),
        controls=ctrl,
        workspace=work,
        classical_bits=bits,
        adjoint=adjoint,
    )
    reference = (source.adjoint() if adjoint else source).decompose_bloq()
    occupied = tuple(
        wire for wire in work if any(wire in gate.wires for gate in actual)
    )
    expected = reference_arithmetic_gates(
        reference,
        ports,
        occupied,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(actual, expected)


def test_append_round_trip_example() -> None:
    from examples.algorithms.indexed_component_round_trip import main

    main()
