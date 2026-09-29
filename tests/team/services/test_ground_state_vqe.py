from __future__ import annotations

import json

import pytest

from flagquantum.services import (
    GroundStateAnsatz,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    exact_sector_reference,
    ground_state_vqe_capability,
    run_ground_state_vqe,
)

pytestmark = pytest.mark.unit


def _h2_hamiltonian() -> PauliSum:
    return PauliSum(
        n_qubits=2,
        terms=(
            PauliSumTerm(-1.052373245772859, "II"),
            PauliSumTerm(0.39793742484318045, "ZI"),
            PauliSumTerm(-0.39793742484318045, "IZ"),
            PauliSumTerm(-0.01128010425623538, "ZZ"),
            PauliSumTerm(0.18093119978423156, "XX"),
        ),
    )


def test_provider_vqe_reaches_and_replays_h2_sector_ground_state() -> None:
    request = GroundStateVQERequest(
        hamiltonian=_h2_hamiltonian(),
        fixed_hamming_weight=1,
    )

    result = run_ground_state_vqe(request)
    reference = exact_sector_reference(request.hamiltonian, fixed_hamming_weight=1)

    assert result.status == "completed"
    assert reference.status == "completed"
    assert result.energy == pytest.approx(-1.85727503, abs=2e-5)
    assert result.energy == pytest.approx(reference.energy, abs=2e-5)
    assert sum(item.contribution for item in result.term_evidence) == pytest.approx(
        result.energy, abs=2e-6
    )
    assert result.statevector is not None
    assert result.statevector.norm == pytest.approx(1.0, abs=1e-6)
    assert result.resources is not None
    assert result.resources.objective_evaluations == 121
    assert len(result.convergence_trace) == 121


def test_provider_identity_and_digest_are_provider_owned_and_repeatable() -> None:
    request = GroundStateVQERequest(
        hamiltonian=_h2_hamiltonian(), fixed_hamming_weight=1
    )

    first = run_ground_state_vqe(request)
    second = run_ground_state_vqe(request)

    assert first.provenance.provider == "FlagQuantum"
    assert first.provenance.package == "flagquantum"
    assert first.provenance.package_version
    assert first.provenance.tool == "flagquantum.services.run_ground_state_vqe"
    assert first.provenance.tool_version == "1.0.0"
    assert first.provenance.execution_digest.startswith("sha256:")
    assert first.provenance.execution_digest == second.provenance.execution_digest
    json.dumps(first.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("case", "code"),
    [
        (
            GroundStateVQERequest(
                hamiltonian=_h2_hamiltonian(),
                fixed_hamming_weight=0,
            ),
            "UNSUPPORTED_SECTOR",
        ),
        (
            GroundStateVQERequest(
                hamiltonian=_h2_hamiltonian(),
                fixed_hamming_weight=1,
                ansatz=GroundStateAnsatz(name="hardware_efficient"),
            ),
            "UNSUPPORTED_ANSATZ",
        ),
        (
            GroundStateVQERequest(
                hamiltonian=PauliSum(
                    n_qubits=2,
                    terms=(PauliSumTerm(1.0, "AB"),),
                ),
                fixed_hamming_weight=1,
            ),
            "UNSUPPORTED_PAULI_TERM",
        ),
    ],
)
def test_unsupported_requests_fail_closed_without_fallback(
    case: GroundStateVQERequest, code: str
) -> None:
    result = run_ground_state_vqe(case)

    assert result.status == "unsupported"
    assert result.energy is None
    assert result.parameters == ()
    assert result.unsupported is not None
    assert result.unsupported.code == code
    assert result.provenance.provider == "FlagQuantum"


def test_exact_reference_is_a_separate_tool_and_evidence_identity() -> None:
    reference = exact_sector_reference(_h2_hamiltonian(), fixed_hamming_weight=1)

    assert reference.status == "completed"
    assert reference.energy == pytest.approx(-1.85727503, abs=1e-8)
    assert reference.sector_basis_states == ("01", "10")
    assert reference.provenance.tool == ("flagquantum.services.exact_sector_reference")
    assert reference.provenance.execution_digest.startswith("sha256:")


def test_capability_discovery_declares_routable_scope_and_no_fallback() -> None:
    capability = ground_state_vqe_capability()

    assert capability["available"] is True
    assert capability["provider"] == "FlagQuantum"
    assert capability["tool_version"] == "1.0.0"
    assert capability["supported"]["n_qubits"] == (2,)
    assert capability["fallback"] is False
