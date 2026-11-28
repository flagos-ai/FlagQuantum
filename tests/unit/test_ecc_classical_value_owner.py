"""Run complete source callbacks where physical Montgomery construction refuses."""

from importlib import import_module
from itertools import product
from typing import Any

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("qualtran")
    import_module("cirq")
    import_module("kickmix")
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.algorithms.primitives.elliptic_curve_addition import (
    EllipticCurveAddition,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.simulation.elliptic_curve_values import (
    addition_cleanup_values,
    addition_coordinates_values,
    addition_finish_values,
    addition_flags_values,
    addition_products_values,
    addition_slope_values,
    affine_point_add_values,
    elliptic_curve_addition_values,
)
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,modulus", ((2, 3), (3, 5), (3, 7), (3, 3)))
def test_all_curve_pairs_and_seven_actual_callbacks_without_physical_admission(
    width: int,
    modulus: int,
) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.cryptography.ecc import ec_add

    points = [
        (0, 0),
        *(
            (x, y)
            for x, y in product(range(modulus), repeat=2)
            if (y * y - x**3 - x - 1) % modulus == 0
        ),
    ]
    dtype = QMontgomeryUInt(width, modulus)
    for (a, b), (x, y) in product(points, repeat=2):
        tangent = (3 * a * a + 1) * pow(2 * b, -1, modulus) % modulus if b else 0
        values = dict(
            zip(
                ("a", "b", "x", "y", "lam_r"),
                (dtype.uint_to_montgomery(v) for v in (a, b, x, y, tangent)),
                strict=True,
            )
        )
        expected = ec_add.ECAdd(width, modulus).on_classical_vals(**values)
        assert elliptic_curve_addition_values(width, modulus, **values) == tuple(
            expected.values()
        )
        callbacks = (
            ("One", lambda **kw: addition_flags_values(modulus, **kw)),
            ("Two", lambda **kw: addition_slope_values(width, modulus, **kw)),
            ("Three", lambda **kw: addition_coordinates_values(modulus, **kw)),
            ("Four", lambda **kw: addition_products_values(width, modulus, **kw)),
            ("Five", lambda **kw: addition_cleanup_values(modulus, **kw)),
            ("Six", addition_finish_values),
        )
        for suffix, callback in callbacks:
            cls = getattr(ec_add, "_ECAddStep" + suffix)
            source = cls(width, modulus)
            args = {r.name: values[r.name] for r in source.signature.lefts()}
            output = source.on_classical_vals(**args)
            assert callback(**args) == tuple(
                output[r.name] for r in source.signature.rights()
            )
            values.update(output)
    if 2 * modulus - 1 >= 1 << width:
        with pytest.raises(ValueError, match="2p-1 to fit"):
            EllipticCurveAddition(width, modulus)


def test_point_object_aliases_and_inconsistent_curve_diagnostic_are_unchanged() -> None:
    point = ECPoint(0, 1, 3, 1)
    zero = ECPoint.inf(3, 1)
    assert zero + point is point
    assert point + zero is point
    assert point + (-point) == zero
    with pytest.raises(ValueError, match="consistent mod and curve"):
        point + ECPoint(0, 1, 5, 1)


@pytest.mark.parametrize("width,modulus", ((2, 3), (3, 5)))
def test_values_and_complete_staged_callback_execute_without_source_sdks(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    modulus: int,
) -> None:
    import sys

    for name in tuple(sys.modules):
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}:
            monkeypatch.setitem(sys.modules, name, None)
    assert elliptic_curve_addition_values(width, modulus, 0, 0, 0, 0, 0) == (
        0,
        0,
        0,
        0,
        0,
    )
    assert affine_point_add_values((0, 0), (0, 1), modulus, 1) == (0, 1)


def test_noninvertible_numerical_slope_preserves_original_exception() -> None:
    try:
        from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepTwo
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    args: dict[str, Any] = dict(f1=0, ctrl=1, a=0, b=0, x=0, y=1, lam_r=0)
    with pytest.raises(ValueError, match="not invertible"):
        _ECAddStepTwo(2, 3).on_classical_vals(**args)
    with pytest.raises(ValueError, match="not invertible"):
        addition_slope_values(2, 3, **args)
