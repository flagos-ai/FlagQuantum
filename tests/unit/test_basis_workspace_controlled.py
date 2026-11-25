"""Controlled gates preserve independently checked basis-workspace fibers."""

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.statevector.basis_workspace import compact_workspace_state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("control", [0, 3])
@pytest.mark.parametrize("name,angle", [("ch", None), ("crz", 0.37), ("crz", -1.13)])
def test_controlled_gate_full_state_matches_dense_reference(
    control: int,
    name: str,
    angle: float | None,
) -> None:
    circuit = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2)
    circuit.rz(0, 0.23).rx(2, 0.41)
    if control == 3:
        circuit.cx(0, 3)
    if angle is None:
        circuit.gate(name, (control, 2))
    else:
        circuit.gate(name, (control, 2), theta=angle)
    if control == 3:
        circuit.cx(0, 3)
    circuit.rz(0, -0.17).h(0)
    source = circuit.to_ir()
    checked = replace(source, metadata={"construction_basis_workspace_prefix": 3})
    actual = compact_workspace_state(checked)
    assert actual is not None
    torch.testing.assert_close(
        actual.flatten(), circuit.state().flatten(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "kind", ["dirty", "quantum_scratch", "target_dependent_control"]
)
def test_unproved_workspace_labels_refuse(kind: str) -> None:
    circuit = fq.Circuit(4, dtype=torch.complex128).h(0).h(2)
    if kind == "dirty":
        circuit.cx(0, 3).gate("ch", (3, 2))
    elif kind == "quantum_scratch":
        circuit.gate("ch", (0, 3))
    else:
        circuit.cx(2, 3).gate("ch", (3, 2)).cx(2, 3)
    source = replace(
        circuit.to_ir(), metadata={"construction_basis_workspace_prefix": 3}
    )
    assert compact_workspace_state(source) is None


def test_conditional_rz_scalar_phase_interferes_on_control() -> None:
    circuit = fq.Circuit(3, dtype=torch.complex128).h(0)
    circuit.gate("crz", (0, 1), theta=1.17).h(0)
    source = replace(
        circuit.to_ir(), metadata={"construction_basis_workspace_prefix": 2}
    )
    actual = compact_workspace_state(source)
    assert actual is not None
    torch.testing.assert_close(
        actual.flatten(), circuit.state().flatten(), atol=1e-12, rtol=0
    )
