"""Complete frozen root initializer exports and defining-object identities."""

import pytest

from tests.unit.test_arithmetic_export_file import (
    test_complete_source_export_order_and_definition_identity as assert_source_exports,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXPORTS = (
    "Bloq",
    "DecomposeTypeError",
    "DecomposeNotImplementedError",
    "BloqError",
    "CompositeBloq",
    "BloqBuilder",
    "DidNotFlattenAnythingError",
    "SoquetT",
    "ConnectionT",
    "QCDType",
    "CDType",
    "QDType",
    "QAny",
    "QBit",
    "CBit",
    "QInt",
    "CInt",
    "QIntOnesComp",
    "CIntOnesComp",
    "QUInt",
    "CUInt",
    "BQUInt",
    "BCUInt",
    "QFxp",
    "CFxp",
    "QMontgomeryUInt",
    "CMontgomeryUInt",
    "QGF",
    "CGF",
    "QGFPoly",
    "CGFPoly",
    "QDTypeCheckingSeverity",
    "check_dtypes_consistent",
    "Register",
    "Signature",
    "Side",
    "BloqInstance",
    "Connection",
    "DanglingT",
    "LeftDangle",
    "RightDangle",
    "Soquet",
    "GateWithRegisters",
    "Adjoint",
    "Controlled",
    "CtrlSpec",
    "AddControlledT",
    "_ControlledBase",
    "make_ctrl_system_with_correct_metabloq",
    "BloqExample",
    "bloq_example",
    "BloqDocSpec",
)


def test_complete_root_initializer() -> None:
    """Check all 52 original exports against their immutable source owners."""
    assert_source_exports(
        "qualtran", "a0f2b5e2d219176be9c7a0017968e6cb25c0cb98", EXPORTS
    )
