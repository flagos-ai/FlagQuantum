"""Compare indexed children and target-position mapping with untouched source."""

from collections import Counter
from itertools import product
from math import prod
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.indexed_component import (
    indexed_component_ir,
    indexed_component_workspace,
    indexed_pauli_targets_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit
Pauli = Literal["i", "x", "y", "z"]


def _layout(widths: tuple[int, ...], controls: int, target_size: int) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    start = controls
    axes = []
    for width in widths:
        axes.append(tuple(range(start, start + width)))
        start += width
    targets = tuple(range(start, start + target_size))
    count = indexed_component_workspace(widths, controls)
    work = tuple(range(start + target_size, start + target_size + count))
    return (
        tuple(range(controls)),
        tuple(axes),
        targets,
        work,
        tuple(range(40, 40 + count)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths,controls,position",
    [
        (shape, widths, controls, position)
        for shape, widths, controls in (
            ((2,), (1,), 0),
            ((2,), (1,), 1),
            ((2,), (1,), 2),
            ((2,), (1,), 3),
            ((3,), (2,), 1),
            ((3,), (4,), 0),
            ((3,), (4,), 1),
            ((2, 2), (1, 1), 0),
            ((2, 2), (1, 1), 1),
            ((2, 2), (1, 1), 2),
            ((2, 3), (1, 2), 1),
            ((1, 2), (1, 1), 1),
        )
        for position in (False, True)
        if position or controls <= 1
    ],
)
@pytest.mark.parametrize("pattern", ["xz", "iy"])
def test_source_complete_gate_streams_and_resource_categories(
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
    registers = tuple(
        Register(f"s{index}", BQUInt(width, extent))
        for index, (width, extent) in enumerate(zip(widths, shape, strict=True))
    )
    ports = {
        register.name: axis for register, axis in zip(registers, axes, strict=True)
    }
    if controls:
        ports["control"] = ctrl
    if position:

        def nth(*indices: int):
            flat = int(np.ravel_multi_index(indices, shape))
            return {"i": cirq.I, "x": cirq.X, "y": cirq.Y, "z": cirq.Z}[labels[flat]]

        source = ApplyGateToLthQubit(
            registers,
            nth,
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
            selection_regs=registers,
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
        )
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(actual, expected)
    categories = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in actual)
    assert counts["t"] + counts["tdg"] == categories.total_t_count()
    assert counts["measure"] == categories.measurement
    used_work = {wire for gate in actual for wire in gate.wires if wire in work}
    assert get_cost_value(source, QubitCount()) == len(ctrl) + sum(widths) + len(
        targets
    ) + len(used_work)


@pytest.mark.parametrize("position", [False, True])
@pytest.mark.parametrize(
    "shape,widths,controls",
    [
        ((2,), (1,), 0),
        ((2,), (1,), 1),
        ((3,), (2,), 1),
        ((2, 2), (1, 1), 0),
        ((2,), (1,), 2),
        ((2,), (1,), 3),
    ],
)
@pytest.mark.parametrize("pattern", ["xz", "iy"])
def test_all_small_complex_branches_and_clean_work(
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    controls: int,
    position: bool,
    pattern: str,
) -> None:
    ctrl, axes, targets, work, bits = _layout(
        widths, controls, prod(shape) if position else 1
    )
    labels: tuple[Pauli, ...] = tuple(
        pattern[index % len(pattern)] for index in range(prod(shape))
    )
    if position:
        gates = indexed_pauli_targets_ir(
            axes,
            shape,
            targets,
            labels,
            controls=ctrl,
            workspace=work,
            classical_bits=bits,
        )
    else:
        gates = indexed_component_ir(
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
        )
    data = len(ctrl) + sum(widths) + len(targets)
    total = data + len(work)
    generator = torch.Generator().manual_seed(53)
    amplitudes = torch.complex(
        torch.randn(1 << data, generator=generator, dtype=torch.float64),
        torch.randn(1 << data, generator=generator, dtype=torch.float64),
    )
    # Source's non-power-of-two padding aliases are tested against source streams;
    # the independent SELECT formula applies on the declared address domain only.
    expected = torch.zeros(1 << total, dtype=torch.complex128)
    initial = torch.zeros_like(expected)
    for word in range(1 << data):
        address = []
        for axis in axes:
            value = sum(
                ((word >> (data - 1 - wire)) & 1) << (len(axis) - 1 - index)
                for index, wire in enumerate(axis)
            )
            address.append(value)
        if any(value >= extent for value, extent in zip(address, shape, strict=True)):
            continue
        flat = 0
        for value, extent in zip(address, shape, strict=True):
            flat = flat * extent + value
        active = all((word >> (data - 1 - wire)) & 1 for wire in ctrl)
        updated, sign = word, 1
        if active:
            target = targets[flat] if position else targets[0]
            mask = 1 << (data - 1 - target)
            if labels[flat] == "x":
                updated ^= mask
            elif labels[flat] == "y":
                updated ^= mask
                sign = -1j if word & mask else 1j
            elif labels[flat] == "z" and word & mask:
                sign = -1
        initial[word << len(work)] = amplitudes[word]
        expected[updated << len(work)] = sign * amplitudes[word]
    norm = initial.norm()
    initial /= norm
    expected /= norm
    events = sum(gate.name == "measure" for gate in gates)
    for outcomes in product((0, 1), repeat=events):
        branch = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            branch, expected / (2 ** (events / 2)), atol=2e-13, rtol=2e-13
        )
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(total, gates)
    )
    result = run_dynamic(circuit, shots=8, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-13,
        rtol=2e-13,
    )


def test_invalid_binding_fails_before_callback() -> None:
    def unexpected(index: int, predicate: int) -> tuple[Instruction, ...]:
        pytest.fail("Invalid parent bindings must not call the child")

    for axes, shape, controls, work, bits in [
        (((),), (1,), (), (), ()),
        (((0,),), (1,), (), (), ()),
        (((0,),), (3,), (), (), ()),
        (((0,),), (2,), (), (1,), (0,)),
        (((0, 1),), (3,), (2,), (3, 4), (0, 0)),
        (((0, 1),), (3,), (2,), (2, 4), (0, 1)),
    ]:
        with pytest.raises(ValueError):
            indexed_component_ir(
                axes,
                shape,
                unexpected,
                controls=controls,
                workspace=work,
                classical_bits=bits,
            )


@pytest.mark.ftqc_reference
def test_non_pauli_child_retains_source_gate_and_model_distinction() -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ApplyLthBloq(np.array((Hadamard(), XGate()), dtype=object), control_val=1)
    gates = indexed_component_ir(
        ((1,),),
        (2,),
        lambda index, predicate: (
            Instruction("ch" if index == 0 else "cx", (predicate, 2)),
        ),
        controls=(0,),
        workspace=(3,),
        classical_bits=(0,),
    )
    expected = reference_arithmetic_gates(
        source,
        {"control": (0,), "selection": (1,), "q": (2,)},
        (3,),
        {3: 0},
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(gates, expected)
    counts = Counter(gate.name for gate in gates)
    assert counts["t"] + counts["tdg"] == 4
    assert counts["ch"] == 1
    model = get_cost_value(source, QECGatesCost())
    assert model.rotation == 2
    assert model.total_t_count() == 26  # two modeled rotations at11 T, one And at4 T
    initial = torch.zeros(16, dtype=torch.complex128)
    amplitude = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    amplitude /= amplitude.norm()
    initial[::2] = amplitude
    ideal = torch.zeros_like(initial)
    for word in range(8):
        enable, address, target = (word >> 2) & 1, (word >> 1) & 1, word & 1
        if not enable:
            ideal[word << 1] += amplitude[word]
        elif address:
            ideal[(word ^ 1) << 1] += amplitude[word]
        else:
            ideal[(word & ~1) << 1] += amplitude[word] / 2**0.5
            ideal[((word & ~1) | 1) << 1] += (
                (-1 if target else 1) * amplitude[word] / 2**0.5
            )
    for outcome in (0, 1):
        actual = forced_branch(initial, gates, (outcome,))
        reference = forced_branch(initial, expected, (outcome,))
        torch.testing.assert_close(actual, reference, atol=2e-13, rtol=2e-13)
        torch.testing.assert_close(actual, ideal / 2**0.5, atol=2e-13, rtol=2e-13)


@pytest.mark.ftqc_reference
def test_source_zero_control_value_reproducer_is_not_repaired() -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    streams = []
    for value in (0, 1):
        source = ApplyLthBloq(
            np.array((XGate(), ZGate()), dtype=object), control_val=value
        )
        streams.append(
            reference_arithmetic_gates(
                source,
                {"control": (0,), "selection": (1,), "q": (2,)},
                (3,),
                {3: 0},
                cirq_style=True,
                keep_pauli_leaves=True,
            )
        )
    assert streams[0] == streams[1]
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[0] = 1  # control0/address0/target0: a negative X control would flip target
    for outcome in (0, 1):
        torch.testing.assert_close(
            forced_branch(initial, streams[0], (outcome,)),
            initial / 2**0.5,
            atol=2e-13,
            rtol=2e-13,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("position", [False, True])
def test_empty_inner_axis_preserves_original_source_failure(position: bool) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import BQUInt, Register
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    regs = (Register("a", BQUInt(1, 2)), Register("b", BQUInt(0, 1)))
    if position:
        source = ApplyGateToLthQubit(regs, lambda *indices: cirq.X, control_regs=())
        ports = {"a": (0,), "b": (), "target": (1, 2)}
    else:
        source = ApplyLthBloq(
            np.array(((XGate(),), (XGate(),)), dtype=object), selection_regs=regs
        )
        ports = {"a": (0,), "b": (), "q": (1,)}
    with pytest.raises(AssertionError):
        reference_arithmetic_gates(
            source, ports, (), {}, cirq_style=True, keep_pauli_leaves=True
        )
    with pytest.raises(ValueError, match="positive concrete"):
        indexed_component_ir(
            ((0,), ()),
            (2, 1),
            lambda index, predicate: (),
            workspace=(),
            classical_bits=(),
        )


def test_append_example() -> None:
    from examples.algorithms.indexed_component_dispatch import main

    main()
