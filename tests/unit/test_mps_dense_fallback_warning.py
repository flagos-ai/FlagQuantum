"""Rule 3 requires the MPS dense fallback to be reported, not silent.

`CompiledMPSProgram` already labelled every operation with three or more wires
`dense_fallback`, but nothing consumed that label: `MPSState.apply_instruction`
expanded the whole state to a 2**n dense vector and re-factorised it without any
warning, result field, or plan diagnostic. A caller therefore saw an
`recommended_mode="mps"` plan whose `state_bytes` was the dense size, while the
bond dimension that supposedly bounded the run did not bound that step at all.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.mps import entrypoints as mps_entrypoints
from flagquantum.simulation.mps.models import CompiledMPSProgram

pytestmark = pytest.mark.unit


def _circuit(n_wires: int) -> fq.Circuit:
    return fq.Circuit(n_wires).h(0).ccx(0, 1, 2)


def test_three_wire_gate_is_planned_as_dense_fallback() -> None:
    program = CompiledMPSProgram.compile(
        tuple(_circuit(4).to_ir()),
        signature=(4, 1, "cpu", torch.complex64, None, 0.0, True),
        fuse_single_qubit=True,
    )
    assert "dense_fallback" in {operation.kind for operation in program.operations}


def test_local_mps_reports_the_dense_fallback() -> None:
    with pytest.warns(RuntimeWarning, match="dense vector and re-factorising"):
        mps_entrypoints.run_mps(_circuit(6), max_bond=4, cutoff=1e-8)


def test_local_mps_reports_the_fallback_once_per_program() -> None:
    circuit = fq.Circuit(6).ccx(0, 1, 2).ccx(3, 4, 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mps_entrypoints.run_mps(circuit, max_bond=4, cutoff=1e-8)
    fallback_warnings = [
        item for item in caught if issubclass(item.category, RuntimeWarning)
    ]
    assert len(fallback_warnings) == 1


def test_two_wire_only_execution_stays_silent() -> None:
    circuit = fq.Circuit(6).h(0).cx(0, 1).cx(1, 2).cx(4, 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mps_entrypoints.run_mps(circuit, max_bond=4, cutoff=1e-8)
    assert [item for item in caught if issubclass(item.category, RuntimeWarning)] == []


def test_reported_fallback_keeps_the_numerics_exact() -> None:
    """Reporting the fallback must not change the state it reports.

    The independent reference is the statevector backend, whose flat state layout
    matches ``MPSState.to_statevector()`` element for element.
    """

    circuit = _circuit(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        state = mps_entrypoints.run_mps(circuit, max_bond=4, cutoff=1e-12)

    reference = circuit.state()
    assert torch.allclose(
        state.to_statevector().reshape(-1), reference.reshape(-1), atol=1e-5
    )
