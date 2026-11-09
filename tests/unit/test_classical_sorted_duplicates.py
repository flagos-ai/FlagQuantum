"""Sorted duplicate callbacks preserve in-place flags and unused control fields."""

from collections import Counter
from copy import deepcopy
from itertools import combinations_with_replacement, product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_sorted_duplicates import (
    sorted_duplicates_classical_values,
)
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ARRAYS = [
    np.array([], dtype=np.int64),
    np.array([0]),
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([0, 1, 1]),
    np.array([2, 1]),
    np.array([-1, -1, 0]),
    np.array([False, True]),
    np.array([1.0, 1.0]),
    np.array([np.nan]),
    np.array(1),
    [0, 0],
    np.array([[0, 0], [1, 2]]),
    np.array([[0, 1], [0, 2]]),
    np.array([[1, 0], [0, 2]]),
    np.empty((0, 2)),
    np.array([0, 0], dtype=object),
    np.array([sympy.Integer(0), sympy.Integer(0)], dtype=object),
]
FLAGS = [
    0,
    1,
    3,
    -1,
    True,
    np.bool_(True),
    np.uint8(255),
    np.array(0),
    np.array([0, 1]),
    np.array([True, False]),
    None,
    1.0,
    np.array([0.0, 1.0]),
]


def source(length=2, width=3, signed=False, controlled=False):
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    return HasDuplicates(length, (QInt if signed else QUInt)(width), controlled)


def compare_registers(original, registers):
    original_inputs, native_inputs = deepcopy(registers), deepcopy(registers)
    try:
        expected = original.on_classical_vals(**original_inputs)
    except Exception as error:
        with pytest.raises(type(error)):
            sorted_duplicates_classical_values(native_inputs)
    else:
        actual = sorted_duplicates_classical_values(native_inputs)
        assert actual is not native_inputs and type(actual) is type(expected) is dict
        assert tuple(actual) == tuple(expected)
        for name in actual:
            same_value(actual[name], expected[name])
    # Source **kwargs is a shallow copy. In-place ndarray changes remain visible
    # in caller-owned input mappings even when a later operation raises.
    assert tuple(native_inputs) == tuple(original_inputs)
    for name in original_inputs:
        same_value(native_inputs[name], original_inputs[name])


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("xs", ARRAYS)
@pytest.mark.parametrize("flag", FLAGS)
def test_original_sort_assertion_adjacency_flag_types_and_mutation(
    controlled, xs, flag
):
    compare_registers(
        source(controlled=controlled), {"xs": xs, "flag": flag, "ctrl": 0}
    )


@pytest.mark.parametrize(
    "length,width",
    [
        (0, 0),
        (1, 3),
        (5, 3),
        (-1, -1),
        (sympy.Symbol("l"), sympy.Symbol("n")),
        (np.int64(2), sympy.Integer(3)),
    ],
)
@pytest.mark.parametrize("signed,controlled", list(product((False, True), repeat=2)))
def test_unused_signature_metadata_does_not_change_raw_callback(
    length, width, signed, controlled
):
    compare_registers(
        source(length, width, signed, controlled),
        {"xs": np.array([0, 0, 1]), "flag": 0, "ctrl": 0, "extra": np.array([9])},
    )


@pytest.mark.parametrize(
    "registers", [{}, {"flag": 0}, {"xs": np.array([0, 1])}, {"xs": np.array([0, 0])}]
)
def test_missing_keys_follow_source_evaluation_order(registers):
    compare_registers(source(), registers)


def test_shallow_dictionary_copy_preserves_original_xs_and_flag_aliases():
    xs, flag, extra = [0, 0], np.array([0, 1]), object()
    registers = {"xs": xs, "flag": flag, "extra": extra}
    result = sorted_duplicates_classical_values(registers)
    assert result is not registers
    assert result["xs"] is xs and result["flag"] is flag and result["extra"] is extra
    np.testing.assert_array_equal(flag, [1, 0])
    assert xs == [0, 0]
    unchanged = np.array([0, 1])
    result = sorted_duplicates_classical_values({"xs": unchanged, "flag": flag})
    assert result["xs"] is unchanged and result["flag"] is flag
    np.testing.assert_array_equal(flag, [1, 0])


def test_multidimensional_adjacency_and_sortedness_are_not_flattened():
    # Original sorting is along the last axis; adjacency is between rows.
    xs = np.array([[0, 0], [1, 2]])
    assert sorted_duplicates_classical_values({"xs": xs, "flag": 0})["flag"] == 0
    xs = np.array([[0, 1], [0, 2]])
    assert sorted_duplicates_classical_values({"xs": xs, "flag": 0})["flag"] == 1
    with pytest.raises(AssertionError):
        sorted_duplicates_classical_values({"xs": np.array([2, 1]), "flag": None})


def test_flag_array_alias_can_change_xs_after_the_original_sortedness_assertion():
    original_data, native_data = np.array([0, 1, 1]), np.array([0, 1, 1])
    expected = source().on_classical_vals(xs=original_data, flag=original_data)
    actual = sorted_duplicates_classical_values(
        {"xs": native_data, "flag": native_data}
    )
    assert actual["xs"] is actual["flag"] is native_data
    assert expected["xs"] is expected["flag"] is original_data
    np.testing.assert_array_equal(native_data, original_data)
    np.testing.assert_array_equal(native_data, [1, 0, 0])


@pytest.mark.parametrize("signed,controlled", list(product((False, True), repeat=2)))
def test_original_callback_values_drive_active_complex_channel_and_clean_work(
    signed, controlled
):
    import torch
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.sorting.sorted_duplicates import (
        sorted_duplicate_requirements,
        sorted_duplicates_ir,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_sorted_duplicates import layout

    words, flag, work, slots, control, total = layout(2, 1, controlled)
    gates, outputs = sorted_duplicates_ir(
        words, flag, work, slots, signed=signed, control=control, source_aliases=True
    )
    assert outputs == tuple(reversed(words))
    domain = (-1, 0) if signed else (0, 1)
    states = list(product(combinations_with_replacement(domain, 2), (0, 1), (0, 1)))
    grid = torch.arange(1, len(states) + 1, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(1 << (total + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for index, (values, target, spectator) in enumerate(states):
        fields = {"xs": np.array(values), "flag": target}
        if controlled:
            fields["ctrl"] = 1
        output = sorted_duplicates_classical_values(fields)
        data = ((values[0] & 1) << 2) | ((values[1] & 1) << 1) | target
        updated = (data & ~1) | int(output["flag"])
        if controlled:
            data, updated = (data << 1) | 1, (updated << 1) | 1
        initial[(data << (len(work) + 1)) | spectator] = weights[index]
        expected[(updated << (len(work) + 1)) | spectator] = weights[index]
    measurements = Counter(g.name for g in gates)["measure"]
    for history in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (measurements / 2), rtol=0, atol=2e-12
        )
        assert (
            torch.count_nonzero(
                actual.reshape(1 << (total - len(work)), 1 << len(work), 2)[:, 1:, :]
            )
            == 0
        )
    model = get_cost_value(source(2, 1, signed, controlled), QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert measurements == model.measurement
    assert sorted_duplicate_requirements(2, 1, controlled=controlled)[0] == len(work)


def test_original_controlled_callback_ignores_control_but_decomposition_does_not():
    import torch

    from flagquantum.compiler.registers.sorting.sorted_duplicates import (
        sorted_duplicates_ir,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_sorted_duplicates import layout

    original = source(2, 2, controlled=True)
    fields = {"xs": np.array([1, 1], dtype=np.uint8), "flag": 0, "ctrl": 0}
    assert (
        original.on_classical_vals(**fields)["flag"]
        == sorted_duplicates_classical_values(fields)["flag"]
        == 1
    )
    decomposed = original.decompose_bloq().call_classically(**fields)
    assert decomposed[1] == 0
    np.testing.assert_array_equal(decomposed[0], fields["xs"])
    words, flag, work, slots, control, total = layout(2, 2, True)
    gates, _ = sorted_duplicates_ir(
        words, flag, work, slots, control=control, source_aliases=True
    )
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[20 << len(work)] = 1j  # Two raw 01 words, flag=0, ctrl=0.
    measurements = Counter(g.name for g in gates)["measure"]
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            initial / 2 ** (measurements / 2),
            rtol=0,
            atol=2e-12,
        )
