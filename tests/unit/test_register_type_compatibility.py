"""Pinned consistency decisions over native quantum and classical metadata."""

from enum import Enum

import pytest
import sympy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
N = sympy.Symbol("n", integer=True, positive=True)


class Severity(Enum):
    LOOSE = 0
    ANY = 1
    STRICT = 2


@pytest.fixture(scope="module")
def register_cases():
    import qualtran as source

    from flagquantum.core.ir.program import register_types as native
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    cases = []
    factories = [
        (source.QAny, native.opaque_register_type),
        (source.QUInt, native.unsigned_register_type),
        (source.QInt, native.signed_register_type),
        (source.QIntOnesComp, native.ones_complement_register_type),
        (source.QMontgomeryUInt, native.montgomery_register_type),
        (source.CUInt, native.classical_unsigned_register_type),
        (source.CInt, native.classical_signed_register_type),
        (source.CIntOnesComp, native.classical_ones_complement_register_type),
        (source.CMontgomeryUInt, native.classical_montgomery_register_type),
    ]
    for original, replacement in factories:
        for width in (0, 1, 3, 4, N):
            if original in (source.QIntOnesComp, source.CIntOnesComp) and width == 1:
                continue
            cases.append((original(width), replacement(width)))
    cases.extend(
        [
            (source.QBit(), native.quantum_bit_type()),
            (source.CBit(), native.classical_bit_type()),
        ]
    )
    for width in (1, 3, 4, N):
        cases.extend(
            [
                (
                    source.BQUInt(width, 1),
                    native.bounded_unsigned_register_type(width, 1),
                ),
                (
                    source.BCUInt(width, 1),
                    native.classical_bounded_unsigned_register_type(width, 1),
                ),
                (
                    source.QMontgomeryUInt(width, 3),
                    native.montgomery_register_type(width, 3),
                ),
                (
                    source.CMontgomeryUInt(width, 3),
                    native.classical_montgomery_register_type(width, 3),
                ),
            ]
        )
        for fractional in (0, width):
            cases.extend(
                [
                    (
                        source.QFxp(width, fractional),
                        native.fixed_point_register_type(width, fractional),
                    ),
                    (
                        source.CFxp(width, fractional),
                        native.classical_fixed_point_register_type(width, fractional),
                    ),
                ]
            )
    for width in (3, 4, N):
        for signed in (False, True):
            cases.extend(
                [
                    (
                        source.QFxp(width, 1, signed),
                        native.fixed_point_register_type(width, 1, signed),
                    ),
                    (
                        source.CFxp(width, 1, signed),
                        native.classical_fixed_point_register_type(width, 1, signed),
                    ),
                ]
            )
    for characteristic, degree in ((2, 1), (2, 2), (2, 3), (3, 2), (2, N)):
        for classical in (False, True):
            for explicit_none in (False, True):
                constructor = source.CGF if classical else source.QGF
                reference = (
                    constructor(characteristic, degree, None)
                    if explicit_none
                    else constructor(characteristic, degree)
                )
                field = (
                    FiniteFieldEncoding(characteristic, degree, None)
                    if explicit_none
                    else FiniteFieldEncoding(characteristic, degree)
                )
                cases.append((reference, field.register_type(classical=classical)))
    return cases


def reference_check(left, right, severity):
    from qualtran._infra.data_types import (
        QDTypeCheckingSeverity,
        check_dtypes_consistent,
    )

    return check_dtypes_consistent(
        left[0], right[0], QDTypeCheckingSeverity[severity.name]
    )


def native_check(left, right, severity):
    from flagquantum.core.ir.program.register_compatibility import (
        RegisterTypeCheckingSeverity,
        check_register_types_consistent,
    )

    return check_register_types_consistent(
        left[1], right[1], RegisterTypeCheckingSeverity[severity.name]
    )


@pytest.fixture(params=["source", "native"])
def check(request):
    return reference_check if request.param == "source" else native_check


def outcome(call):
    try:
        result = call()
    except (TypeError, ValueError) as error:
        return type(error), str(error)
    return type(result), result


@pytest.mark.parametrize("severity", list(Severity))
def test_supported_type_cross_product(check, register_cases, severity):
    comparisons = 0
    for left in register_cases:
        for right in register_cases:
            assert outcome(lambda: check(left, right, severity)) == outcome(
                lambda: reference_check(left, right, severity)
            ), (left[0], right[0], severity)
            comparisons += 1
    assert comparisons == len(register_cases) ** 2
