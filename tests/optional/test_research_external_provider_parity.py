"""Real-dependency parity for replaceable local research providers."""

from __future__ import annotations

import importlib.util

import pytest

from flagquantum.services import (
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    execute_research_capability,
    validate_research_evidence,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        any(
            importlib.util.find_spec(name) is None
            for name in ("qiskit_aer", "pennylane", "pennylane_lightning")
        ),
        reason="Qiskit Aer and PennyLane Lightning are both required",
    ),
]


def _request() -> GroundStateVQERequest:
    return GroundStateVQERequest(
        hamiltonian=PauliSum(
            n_qubits=2,
            terms=(
                PauliSumTerm(-1.052373245772859, "II"),
                PauliSumTerm(0.39793742484318045, "ZI"),
                PauliSumTerm(-0.39793742484318045, "IZ"),
                PauliSumTerm(-0.01128010425623538, "ZZ"),
                PauliSumTerm(0.18093119978423156, "XX"),
            ),
        ),
        fixed_hamming_weight=1,
    )


def test_qiskit_and_pennylane_satisfy_the_same_scientific_contract() -> None:
    request = _request()
    evidence = {
        provider: execute_research_capability(
            "ground_state.solve.vqe",
            request,
            {"provider_order": [provider]},
        )
        for provider in ("qiskit-aer", "pennylane-lightning")
    }

    for provider, bundle in evidence.items():
        assert bundle.status == "succeeded"
        assert bundle.capability.provider == provider
        assert bundle.result["resources"]["objective_evaluations"] == 121
        assert bundle.result["resources"]["backend_executions"] == 361
        assert validate_research_evidence("validate.energy_variance", bundle).passed
    assert evidence["qiskit-aer"].result["energy"] == pytest.approx(
        evidence["pennylane-lightning"].result["energy"], abs=1e-6
    )
    assert evidence["qiskit-aer"].metrics["exact_energy_error"] == pytest.approx(
        evidence["pennylane-lightning"].metrics["exact_energy_error"], abs=1e-6
    )
