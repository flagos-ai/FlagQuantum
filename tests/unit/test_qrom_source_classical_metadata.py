"""Host-value evaluation preserves deferred QROM metadata and control bypasses."""

from copy import deepcopy
from importlib import import_module
from typing import Any

import pytest

from flagquantum.compiler.registers.lookup.qrom_declaration import QromDeclaration
from flagquantum.simulation.classical_lookup import source_lookup_classical_values
from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    import sympy

    QROM = import_module("qualtran.bloqs.data_loading.qrom").QROM
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controls", (-2, 0, True, 2, sympy.Integer(1)))
@pytest.mark.parametrize("active", (False, True))
def test_host_values_match_source_control_metadata(controls: Any, active: bool) -> None:
    arrays = (np.arange(4).reshape(2, 2),)
    source = QROM(arrays, (1,), (2,), num_controls=controls)
    declaration = QromDeclaration.from_arrays(
        *arrays, selection_bitsizes=(1,), target_bitsizes=(2,), num_controls=controls
    )
    values = {"selection": 1, "target0_": np.array([1, 2]), "unused": 9}
    if controls > 0:
        values["control"] = 2**controls - 1 if active else 0
    before = deepcopy(values)
    actual = source_lookup_classical_values(
        arrays, declaration.selection_axes, values, num_controls=controls
    )
    expected = source.on_classical_vals(**values)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert type(actual[key]) is type(expected[key])
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in values:
        np.testing.assert_array_equal(values[key], before[key])


def test_inactive_control_bypasses_inconsistent_axes_and_checked_erasure() -> None:
    arrays = (np.array([1, 2]), np.array([1, 2, 3]))
    source = QROM(arrays, (1,), (2, 2), num_controls=1)
    declaration = QromDeclaration.from_arrays(
        *arrays, selection_bitsizes=(1,), target_bitsizes=(2, 2), num_controls=1
    )
    values = {"control": 0, "target0_": 99, "unused": sympy.Symbol("x")}
    for erase in (False, True):
        assert source_lookup_classical_values(
            arrays, declaration.selection_axes, values, num_controls=1, erase=erase
        ) == source.on_classical_vals(**values)
    with pytest.raises(ValueError, match="same shape"):
        source.on_classical_vals(control=1)
    with pytest.raises(ValueError, match="same shape"):
        source_lookup_classical_values(
            arrays, declaration.selection_axes, {"control": 1}, num_controls=1
        )


def test_unknown_data_refuses_before_control_bypass() -> None:
    source = QROM.build_from_bitsize(4, 2, num_controls=1)
    declaration = QromDeclaration.from_shape(4, 2, num_controls=1)
    with pytest.raises(NotImplementedError):
        source.on_classical_vals(control=0)
    with pytest.raises(NotImplementedError):
        source_lookup_classical_values(
            None, declaration.selection_axes, {"control": 0}, num_controls=1
        )


def test_host_values_do_not_request_invalid_target_storage_metadata() -> None:
    arrays = (np.arange(4).reshape(2, 2),)
    targets: Any = ((1.5,),)
    source = QROM(arrays, (1,), (2,), targets)
    declaration = QromDeclaration.from_arrays(
        *arrays, selection_bitsizes=(1,), target_bitsizes=(2,), target_shapes=targets
    )
    actual = source_lookup_classical_values(
        arrays, declaration.selection_axes, {"selection": 1}
    )
    expected = source.on_classical_vals(selection=1)
    for key in expected:
        assert type(actual[key]) is type(expected[key])
        np.testing.assert_array_equal(actual[key], expected[key])
