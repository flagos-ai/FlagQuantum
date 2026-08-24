from __future__ import annotations

import pytest

from flagquantum.interop.pennylane import run_pennylane_conformance

pytest.importorskip("pennylane")
pytestmark = [pytest.mark.integration, pytest.mark.pennylane]


def test_common_adapter_and_complex128_conformance_pass() -> None:
    result = run_pennylane_conformance()
    assert result.schema == "flagquantum_pennylane_conformance_v1"
    assert result.pennylane_version.startswith(("0.44.1", "0.45.1"))
    assert result.dtype == "complex128"
    assert result.passed
    assert result.adapter_contract.passed
    assert all(case.maximum_absolute_error <= 1e-10 for case in result.cases)
