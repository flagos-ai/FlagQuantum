"""Complete locked bookkeeping initializer order and defining-object identities."""

import pytest

from tests.unit.test_arithmetic_export_file import (
    test_complete_source_export_order_and_definition_identity as assert_source_exports,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXPORTS = (
    "Allocate",
    "Always",
    "ArbitraryClifford",
    "AutoPartition",
    "Cast",
    "Free",
    "Join",
    "Join2",
    "Partition",
    "Split2",
    "Split",
)


def test_complete_bookkeeping_initializer() -> None:
    """Check all imports against the frozen blob and their original owners."""
    assert_source_exports(
        "qualtran.bloqs.bookkeeping",
        "3dd8073aa1f9cb05ff8f812de6c60d6879b49607",
        EXPORTS,
    )
