"""Complete locked basic_gates initializer order and defining-object identities."""

import pytest

from tests.unit.test_arithmetic_export_file import (
    test_complete_source_export_order_and_definition_identity as assert_source_exports,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXPORTS = (
    "CNOT",
    "Discard",
    "DiscardQ",
    "GlobalPhase",
    "CHadamard",
    "Hadamard",
    "Identity",
    "OnEach",
    "Power",
    "CRz",
    "CZPowGate",
    "Rx",
    "Ry",
    "Rz",
    "XPowGate",
    "YPowGate",
    "ZPowGate",
    "SGate",
    "SU2RotationGate",
    "CSwap",
    "Swap",
    "TwoBitCSwap",
    "TwoBitSwap",
    "TGate",
    "Toffoli",
    "MeasureX",
    "MinusEffect",
    "MinusState",
    "PlusEffect",
    "PlusState",
    "XGate",
    "CYGate",
    "YGate",
    "CZ",
    "IntEffect",
    "IntState",
    "MeasureZ",
    "OneEffect",
    "OneState",
    "ZeroEffect",
    "ZeroState",
    "ZGate",
)


def test_complete_basic_gate_initializer() -> None:
    """Check all imports against the frozen blob and their original owners."""
    assert_source_exports(
        "qualtran.bloqs.basic_gates",
        "e093c1d7663e8019904a93a5c565e87c256ed724",
        EXPORTS,
    )
