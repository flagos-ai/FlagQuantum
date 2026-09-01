"""Correctness and contract tests for constrained MPS TEBD."""

from __future__ import annotations

import json
import math

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.integration


def _ising(n_wires: int = 2, coupling: float = 1.0, field: float = 0.7):
    return fq.transverse_field_ising(
        n_wires,
        coupling=coupling,
        field=field,
        periodic=False,
    )


def test_tebd_is_explicitly_experimental_not_stable_root_api() -> None:
    assert "run_tebd" not in fq.__all__
    assert "TEBDResult" not in fq.__all__
    assert "simulation" in fq.experimental.__all__
    assert "run_tebd" in fq.experimental.simulation.__all__
    assert "TEBDResult" not in fq.experimental.simulation.__all__
    assert callable(fq.experimental.simulation.run_tebd)


def test_tebd_plus_x_initial_energy_and_auditable_metadata() -> None:
    result = fq.experimental.simulation.run_tebd(
        _ising(n_wires=4, field=0.7),
        n_wires=4,
        total_time=0.02,
        time_step=0.02,
        max_bond=8,
        cutoff=0.0,
        initial_state="+x",
    )

    assert result.initial_energy == pytest.approx(-4 * 0.7, abs=1e-12)
    assert result.steps == 1
    assert len(result.program_sha256) == 64
    assert result.distribution_semantics == "single_device_fast_path"
    assert result.scalability_claim_allowed is False
    assert result.fallback_used is False
    assert result.maximum_bond_dimension <= 8
    assert max(result.post_normalization_squared_norm_errors) < 1e-12
    assert result.to_dict()["backend"] == "mps_tebd"
    assert result.to_dict()["n_wires"] == 4
    assert result.to_dict()["max_bond"] == 8
    json.dumps(result.to_dict())


def test_two_site_tebd_converges_to_exact_ground_energy() -> None:
    coupling = 1.0
    field = 0.7
    result = fq.experimental.simulation.run_tebd(
        _ising(coupling=coupling, field=field),
        n_wires=2,
        total_time=4.0,
        time_step=0.02,
        max_bond=4,
        cutoff=0.0,
    )
    exact = -math.sqrt(coupling**2 + 4 * field**2)

    assert result.final_energy == pytest.approx(exact, abs=5e-4)
    assert result.final_energy < result.initial_energy
    assert result.cumulative_discarded_weight == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    ("hamiltonian", "kwargs", "match"),
    [
        (_ising(), {"total_time": 0.1, "time_step": 0.03}, "integer multiple"),
        (_ising(), {"evolution": "real_time"}, "imaginary_time"),
        (_ising(), {"order": 1}, "second-order"),
        (_ising(), {"initial_state": "random"}, "Unsupported initial_state"),
        (_ising(), {"dtype": torch.float64}, "dtype"),
        (
            fq.Hamiltonian([fq.pauli_term(1.0, "ZZ", (0, 2))]),
            {"n_wires": 3},
            "adjacent",
        ),
        (
            fq.Hamiltonian([fq.pauli_term(1.0 + 0.1j, "X", (0,))]),
            {},
            "must be real",
        ),
        (
            fq.Hamiltonian([fq.pauli_term(1.0, "XYZ", (0, 1, 2))]),
            {"n_wires": 3},
            "one-site and two-site",
        ),
    ],
)
def test_tebd_rejects_unsupported_or_ambiguous_contracts(
    hamiltonian, kwargs, match
) -> None:
    inputs = {
        "n_wires": 2,
        "total_time": 0.1,
        "time_step": 0.1,
        "max_bond": 4,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fq.experimental.simulation.run_tebd(hamiltonian, **inputs)


def test_tebd_does_not_materialize_a_dense_state(monkeypatch) -> None:
    from flagquantum.simulation.mps_state import MPSState

    def reject_dense_materialization(self):
        raise AssertionError("TEBD must not materialize a statevector")

    monkeypatch.setattr(MPSState, "to_statevector", reject_dense_materialization)
    result = fq.experimental.simulation.run_tebd(
        _ising(n_wires=4),
        n_wires=4,
        total_time=0.04,
        time_step=0.02,
        max_bond=8,
    )

    assert result.steps == 2
    assert result.fallback_used is False


def test_tebd_program_hash_binds_execution_configuration() -> None:
    common = {
        "n_wires": 2,
        "total_time": 0.2,
        "time_step": 0.02,
        "max_bond": 4,
    }
    first = fq.experimental.simulation.run_tebd(_ising(), **common)
    second = fq.experimental.simulation.run_tebd(_ising(), **{**common, "max_bond": 2})

    assert first.program_sha256 != second.program_sha256
    assert first.to_dict()["time_step"] == 0.02
