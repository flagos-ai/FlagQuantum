"""Original QROMBase register values and native lookup channels use the same data."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.qrom_table import QromTable
from flagquantum.simulation.classical_lookup import lookup_classical_values
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def dataset_case(mode):
    if mode == "scalar_targets":
        return (4,), (np.array([1, 3, 0, 2]), np.array([0, 1, 1, 0])), ((), ())
    if mode == "two_axes":
        return (2, 3), (np.arange(6).reshape(2, 3) % 4,), ((),)
    if mode == "grouped":
        return (
            (3,),
            (np.array([[1, 0], [2, 3], [0, 1]]), np.array([0, 1, 2])),
            ((2,), ()),
        )
    if mode == "matrix_target":
        return (3,), (np.arange(12).reshape(3, 2, 2) % 4,), ((2, 2),)
    if mode == "singleton":
        return (1,), (np.array([2]),), ((),)
    if mode == "all_singleton":
        return (1, 1), (np.array([[2]]),), ((),)
    raise AssertionError(mode)


def assert_values_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert type(actual[key]) is type(expected[key]), key
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize(
    "mode,controls,erase",
    tuple(
        product(
            (
                "scalar_targets",
                "two_axes",
                "grouped",
                "matrix_target",
                "singleton",
                "all_singleton",
            ),
            (0, 1, 2),
            (False, True),
        )
    ),
)
def test_original_forward_and_left_target_values(mode, controls, erase):
    from qualtran.bloqs.data_loading.qroam_clean import QROAMCleanAdjoint
    from qualtran.bloqs.data_loading.qrom import QROM

    shape, data, targets = dataset_case(mode)
    forward = QROM.build_from_data(*data, target_shapes=targets, num_controls=controls)
    source = (
        QROAMCleanAdjoint.build_from_data(
            *data,
            target_shapes=targets,
            num_controls=controls,
            log_block_sizes=(0,) * len(shape),
        )
        if erase
        else forward
    )
    selections = tuple(reg.name for reg in forward.selection_registers)
    axes = tuple(range(n) for n in shape if n > 1)
    for control, address, initial_target in product(
        range(1 << controls), product(*axes), (0,) if erase else (0, 1)
    ):
        vals = dict(zip(selections, address, strict=True))
        vals.update(
            {
                f"target{i}_": (
                    np.full(target, initial_target, dtype=int)
                    if target
                    else initial_target
                )
                for i, target in enumerate(targets)
            }
        )
        if controls:
            vals["control"] = control
        if erase:
            vals.update(
                {
                    k: v
                    for k, v in forward.on_classical_vals(**vals).items()
                    if k.startswith("target")
                }
            )
        vals["unused"] = np.array([7])
        before = deepcopy(vals)
        data_before = tuple(a.copy() for a in data)
        expected = source.on_classical_vals(**vals)
        actual = lookup_classical_values(
            data, shape, vals, num_controls=controls, erase=erase
        )
        assert_values_equal(actual, expected)
        assert_values_equal(vals, before)
        for value, original in zip(data, data_before, strict=True):
            np.testing.assert_array_equal(value, original)
        if erase and control == (1 << controls) - 1:
            wrong = dict(vals)
            wrong["target0_"] = wrong["target0_"] ^ 1
            with pytest.raises(ValueError, match="must be uncomputed"):
                source.on_classical_vals(**wrong)
            with pytest.raises(ValueError, match="must be uncomputed"):
                lookup_classical_values(
                    data, shape, wrong, num_controls=controls, erase=True
                )


@pytest.mark.parametrize("erase", [False, True])
def test_original_unknown_data_and_default_single_selector(erase):
    from qualtran.bloqs.data_loading.qroam_clean import QROAMCleanAdjoint
    from qualtran.bloqs.data_loading.qrom import QROM

    cls = QROAMCleanAdjoint if erase else QROM
    unknown = cls.build_from_bitsize(4, 2)
    with pytest.raises(NotImplementedError):
        unknown.on_classical_vals(selection=0)
    with pytest.raises(NotImplementedError):
        lookup_classical_values(None, (4,), {"selection": 0}, erase=erase)
    source = cls.build_from_data([1, 2, 0, 3])
    vals = {"target0_": 1} if erase else {}
    expected = source.on_classical_vals(**vals)
    assert_values_equal(
        lookup_classical_values(source.data, (4,), vals, erase=erase), expected
    )


@pytest.mark.parametrize("selection", [-1, 4])
def test_original_numpy_indexing_boundaries(selection):
    from qualtran.bloqs.data_loading.qrom import QROM

    source = QROM.build_from_data([1, 2, 0, 3])
    vals = {"selection": selection, "target0_": 1}
    if selection == -1:
        assert_values_equal(
            lookup_classical_values(source.data, (4,), vals),
            source.on_classical_vals(**vals),
        )
    else:
        with pytest.raises(IndexError):
            source.on_classical_vals(**vals)
        with pytest.raises(IndexError):
            lookup_classical_values(source.data, (4,), vals)


def test_original_omitted_axes_classical_indexing_is_not_silently_repaired():
    from qualtran.bloqs.data_loading.qrom import QROM

    source = QROM.build_from_data([[0, 1]])
    assert source.data_shape == (1, 2)
    assert_values_equal(
        lookup_classical_values(source.data, (1, 2), {"selection": 0}),
        source.on_classical_vals(selection=0),
    )
    assert isinstance(source.on_classical_vals(selection=0)["target0_"], np.ndarray)
    for evaluate in (
        lambda: source.on_classical_vals(selection=1),
        lambda: lookup_classical_values(source.data, (1, 2), {"selection": 1}),
    ):
        with pytest.raises(IndexError):
            evaluate()
    scalar = QROM.build_from_data(3)
    with pytest.raises(IndexError):
        scalar.on_classical_vals()
    with pytest.raises(IndexError):
        lookup_classical_values(scalar.data, (), {})


def test_missing_multidimensional_selector_and_control_keep_source_failures():
    from qualtran.bloqs.data_loading.qrom import QROM

    source = QROM.build_from_data(np.arange(4).reshape(2, 2), num_controls=1)
    for vals in ({}, {"control": 1, "selection0": 0}):
        with pytest.raises(KeyError):
            source.on_classical_vals(**vals)
        with pytest.raises(KeyError):
            lookup_classical_values(source.data, (2, 2), vals, num_controls=1)
    # Inactive controls bypass missing selectors and erasure checks exactly.
    vals = {"control": 0, "target0_": 3, "unused": 17}
    assert (
        lookup_classical_values(source.data, (2, 2), vals, num_controls=1, erase=True)
        == vals
    )


@pytest.mark.parametrize("controls", [0, 1, 2])
def test_classical_oracle_agrees_with_all_quantum_measurement_histories(controls):
    data = (np.array([1, 3, 0, 2]), np.array([0, 1, 1, 0]))
    table = QromTable.from_data(*data, target_bitsizes=(2, 1), num_controls=controls)
    ports = tuple(range(controls + 5))
    work = tuple(range(len(ports), len(ports) + table.workspace_size))
    gates = table.bind_packed(ports, work, tuple(range(len(work))))
    size, qubits = 1 << len(ports), len(ports) + len(work)
    weights = torch.arange(1, size + 1, dtype=torch.float64)
    coefficients = torch.complex(weights, weights % 7 - 3)
    coefficients /= coefficients.norm()
    initial = torch.zeros(1 << qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for value in range(size):
        control, row, target0, target1 = (
            value >> 5,
            (value >> 3) & 3,
            (value >> 1) & 3,
            value & 1,
        )
        vals = {"selection": row, "target0_": target0, "target1_": target1}
        if controls:
            vals["control"] = control
        result = lookup_classical_values(data, (4,), vals, num_controls=controls)
        out = (
            (control << 5)
            | (int(result["selection"]) << 3)
            | (int(result["target0_"]) << 1)
            | int(result["target1_"])
        )
        initial[value << len(work)] = coefficients[value]
        expected[out << len(work)] = coefficients[value]
    events = sum(g.name == "measure" for g in gates)
    assert events == table.resource_model().gates.measurement
    assert sum(g.name in {"t", "tdg"} for g in gates) == 4 * events
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected * 2 ** (-events / 2), atol=2e-12, rtol=0
        )


@pytest.mark.parametrize(
    "kwargs", [{"num_controls": -1}, {"num_controls": True}, {"erase": 0}]
)
def test_native_evaluation_options_are_explicit(kwargs):
    with pytest.raises(ValueError):
        lookup_classical_values((np.array([0, 1]),), (2,), {}, **kwargs)
