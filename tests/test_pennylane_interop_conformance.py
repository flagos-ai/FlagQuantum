from __future__ import annotations

import pytest
import torch

from flagquantum import Circuit
from flagquantum.ecosystem._semantic_parity import (
    maximum_error_up_to_global_phase,
    reference_state,
    semantic_parity_cases,
)
from flagquantum.ecosystem.pennylane import (
    from_pennylane,
    run_pennylane_conformance,
    to_pennylane,
)

qml = pytest.importorskip("pennylane")
pytestmark = [pytest.mark.integration, pytest.mark.pennylane]


@pytest.mark.parametrize("case", semantic_parity_cases(), ids=lambda case: case.name)
def test_shared_cross_framework_semantics(case) -> None:
    exported = to_pennylane(case.program)
    matrix = torch.as_tensor(
        qml.matrix(exported, wire_order=list(range(case.program.n_wires)))
    ).to(torch.complex128)
    initial = torch.zeros(2**case.program.n_wires, dtype=torch.complex128)
    initial[0] = 1
    external_state = matrix @ initial
    round_trip_state = Circuit.from_ir(
        from_pennylane(exported), dtype=torch.complex128
    ).state()[0]
    expected = reference_state(case)

    assert maximum_error_up_to_global_phase(external_state, expected) <= 1e-10
    assert maximum_error_up_to_global_phase(round_trip_state, expected) <= 1e-10


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
