from __future__ import annotations

import pytest

from flagquantum.ecosystem.pennylane import run_pennylane_conformance

pytest.importorskip("pennylane")
pytestmark = [pytest.mark.integration, pytest.mark.pennylane]


def test_common_adapter_and_complex128_conformance_pass() -> None:
    result = run_pennylane_conformance()

    assert result.schema == "flagquantum_pennylane_conformance_v1"
    assert result.pennylane_version.startswith(("0.44.1", "0.45.1"))
    assert result.dtype == "complex128"
    assert result.passed
    assert result.adapter_contract.passed
    assert [case.name for case in result.cases] == [
        "asymmetric_wire_order",
        "controlled_and_ising",
        "seeded_differential_731",
        "seeded_differential_946",
        "seeded_differential_1212",
        "seeded_differential_1597",
        "seeded_differential_2018",
        "seeded_differential_2371",
    ]
    assert [case.kind for case in result.adapter_contract.cases] == [
        "round_trip"
    ] * 8 + ["rejection"]
    assert all(case.maximum_absolute_error <= 1e-10 for case in result.cases)
