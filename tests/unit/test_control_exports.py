"""Complete locked mcmt initializer order and defining-object identities."""

import pytest

from tests.unit.test_arithmetic_export_file import (
    test_complete_source_export_order_and_definition_identity as assert_source_exports,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXPORTS = (
    "And",
    "MultiAnd",
    "ControlledViaAnd",
    "CtrlSpecAnd",
    "MultiControlX",
    "MultiControlZ",
    "MultiTargetCNOT",
)


def test_complete_control_initializer() -> None:
    """Check all imports against the frozen blob and their original owners."""
    assert_source_exports(
        "qualtran.bloqs.mcmt", "23b8caf45e0b6c83ead160e6cfac5ab3d7d10280", EXPORTS
    )
