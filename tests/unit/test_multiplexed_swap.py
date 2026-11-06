"""Compare exact bounded swaps and every measurement branch to pinned Qualtran."""

from collections import Counter
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.multiplexed_swap import multiplexed_cswap_ir
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit


def source_swap(
    shape: tuple[int, ...], widths: tuple[int, ...], controls: int, word: int
) -> Any:
    from qualtran import BQUInt, QBit, Register
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    return MultiplexedCSwap(
        tuple(
            Register(f"s{i}", BQUInt(width, extent))
            for i, (width, extent) in enumerate(zip(widths, shape, strict=True))
        ),
        word,
        tuple(Register(f"c{i}", QBit()) for i in range(controls)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("shape", "widths", "control_count", "word_width"),
    [
        ((3,), (2,), 0, 2),
        ((2,), (3,), 0, 2),
        ((3,), (2,), 1, 1),
        ((3,), (2,), 2, 1),
        ((2,), (1,), 3, 1),
        ((2, 3), (1, 2), 1, 1),
        ((2, 1), (1, 1), 0, 1),
        ((1,), (1,), 1, 1),
    ],
)
def test_pinned_multiplexed_cswap(
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    control_count: int,
    word_width: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    total = 0

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal total
        start = total
        total += count
        return tuple(range(start, total))

    controls = allocate(control_count)
    selections = tuple(allocate(width) for width in widths)
    targets = tuple(allocate(word_width) for _ in range(prod(shape)))
    output = allocate(word_width)
    data_width = total
    work = allocate(sum(widths) + control_count - 1)
    bits = tuple(range(40, 40 + len(work)))
    actual = multiplexed_cswap_ir(
        controls, selections, shape, targets, output, work, bits
    )
    source = source_swap(shape, widths, control_count, word_width)
    ports = {
        **{f"c{i}": (wire,) for i, wire in enumerate(controls)},
        **{f"s{i}": wires for i, wires in enumerate(selections)},
        "targets": tuple(wire for word in targets for wire in word),
        "output": output,
    }
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
    )
    assert_source_dependencies(actual, expected)

    counts = Counter(gate.name for gate in actual)
    model = get_cost_value(source, QECGatesCost())
    assert model.cswap == prod(shape) * word_width
    assert counts["measure"] == counts["reset"] == model.measurement == model.and_bloq
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert source.t_complexity().t == model.total_t_count(ts_per_cswap=7)
    used_work = {wire for gate in actual for wire in gate.wires}.intersection(work)
    assert get_cost_value(source, QubitCount()) == data_width + len(used_work)

    # Exercise every data basis column, including padded addresses and a spectator.
    words = torch.arange(1 << data_width)
    width_data = (prod(shape) + 1) * word_width
    enable = words >> (width_data + sum(widths))
    address = (words >> width_data) & ((1 << sum(widths)) - 1)
    valid = torch.ones(words.numel(), dtype=torch.bool)
    row = torch.zeros_like(words)
    for axis, (width, extent) in enumerate(zip(widths, shape, strict=True)):
        address_shift = sum(widths[axis + 1 :])
        index = (address >> address_shift) & ((1 << width) - 1)
        valid &= index < extent
        row = row * extent + index
    offset = (prod(shape) - row.clamp(max=prod(shape) - 1)) * word_width
    word_mask = (1 << word_width) - 1
    delta = (words ^ (words >> offset)) & word_mask
    swap = (delta << offset) | delta
    image = torch.where(
        valid & (enable == (1 << control_count) - 1), words ^ swap, words
    )

    indices = (words << (len(work) + 1)).repeat_interleave(2)
    image_indices = (image << (len(work) + 1)).repeat_interleave(2)
    spectator = torch.arange(indices.numel()) & 1
    weights = torch.arange(1, indices.numel() + 1, dtype=torch.float64)
    amplitudes = torch.complex(weights, weights * (weights % 5 - 2))
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (total + 1), dtype=torch.complex128)
    logical = torch.zeros_like(state)
    state[indices | spectator] = amplitudes
    logical[image_indices | spectator] = amplitudes
    valid_indices = (indices | spectator)[valid.repeat_interleave(2)]
    dirty_indices = ((torch.arange(state.numel()) >> 1) & ((1 << len(work)) - 1)) != 0
    for history in product((0, 1), repeat=model.measurement):
        result = forced_branch(state, actual, history)
        torch.testing.assert_close(
            result, forced_branch(state, expected, history), atol=2e-12, rtol=0
        )
        torch.testing.assert_close(
            result[valid_indices],
            logical[valid_indices] * 2 ** (-model.measurement / 2),
            atol=2e-12,
            rtol=0,
        )
        assert torch.count_nonzero(result[dirty_indices]) == 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("shape", "widths", "controls", "error", "message"),
    [
        ((1,), (0,), 1, NotImplementedError, "nonempty source selection"),
        ((2, 1), (1, 0), 1, AssertionError, "nonempty source selection"),
        ((1,), (1,), 0, IndexError, "singleton first axis"),
    ],
)
def test_pinned_undecomposable_axes(
    shape: tuple[int, ...],
    widths: tuple[int, ...],
    controls: int,
    error: type[Exception],
    message: str,
) -> None:
    pytest.importorskip("qualtran")
    source = source_swap(shape, widths, controls, 1)
    with pytest.raises(error):
        source.decompose_bloq()
    with pytest.raises(ValueError, match=message):
        multiplexed_cswap_ir(
            tuple(range(controls)),
            tuple(tuple(range(width)) for width in widths),
            shape,
            ((2,),) * prod(shape),
            (3,),
            (),
            (),
        )
