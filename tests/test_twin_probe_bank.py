"""Scenarios for immutable, rank-qualified Twin probe banks."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, MeasurementNode

pytestmark = pytest.mark.integration


def _identifiable_probes():
    return (
        fq.Circuit(2).cx(0, 1),
        fq.Circuit(2).h(0).cx(0, 1),
        fq.Circuit(2).h(1).cx(0, 1),
    )


def _bank(probes=None):
    return fq.twin.prepare_probe_bank(
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        probes=_identifiable_probes() if probes is None else probes,
        parameter_family="local_pauli_readout",
    )


def _canonical(bank: fq.twin.TwinProbeBank) -> bytes:
    return json.dumps(
        bank.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def test_developer_freezes_an_identifiable_local_probe_bank() -> None:
    bank = _bank()

    assert bank.identifiability_status == "identifiable"
    assert bank.matrix_rank == len(bank.parameter_names) == 5
    assert bank.identifiable_parameters == bank.parameter_names
    assert bank.unidentified_parameters == ()
    assert bank.target == "quafu:Shenglian"
    assert bank.physical_qubits == (20, 27)
    assert bank.directed_couplers == ((20, 27),)
    assert bank.parameter_names == (
        "readout_bit_flip[Q20]",
        "readout_bit_flip[Q27]",
        "single_qubit_pauli[Q20]",
        "single_qubit_pauli[Q27]",
        "two_qubit_pauli[Q20->Q27]",
    )


def test_probe_order_does_not_change_payload_or_identity() -> None:
    probes = _identifiable_probes()

    first = _bank(probes)
    second = _bank(tuple(reversed(probes)))

    assert _canonical(first) == _canonical(second)
    assert first.identity == second.identity


def test_many_probes_do_not_substitute_for_matrix_rank() -> None:
    probes = (
        fq.Circuit(2).h(0),
        fq.Circuit(2).x(0),
        fq.Circuit(2).y(0),
        fq.Circuit(2).z(0),
        fq.Circuit(2).rx(0, theta=0.1),
        fq.Circuit(2).ry(0, theta=0.2),
    )

    bank = _bank(probes)

    assert len(bank.probe_circuit_identities) > len(bank.parameter_names)
    assert bank.identifiability_status == "underdetermined"
    assert bank.matrix_rank < len(bank.parameter_names)
    assert "single_qubit_pauli[Q27]" in bank.unidentified_parameters


def test_probe_support_and_explicit_design_matrix_are_identity_bound() -> None:
    bank = _bank()
    payload = bank.to_dict()

    assert len(bank.design_row_labels) == len(bank.design_matrix)
    assert all(len(row) == len(bank.parameter_names) for row in bank.design_matrix)
    assert tuple(identity for identity, _ in bank.probe_parameter_names) == (
        bank.probe_circuit_identities
    )
    assert payload["directed_couplers"] == [[20, 27]]
    assert payload["schema"] == "flagquantum.twin_probe_bank.v2"
    assert len(payload["ideal_row_probabilities"]) == len(bank.design_matrix)
    assert all(
        sum(probabilities) == pytest.approx(1.0)
        for probabilities in bank.ideal_row_probabilities
    )
    assert payload["matrix_rank"] == 5
    assert payload["identifiability_status"] == "identifiable"


def test_mapping_family_and_probe_identity_mismatches_fail_closed() -> None:
    with pytest.raises(ValueError, match="map every physical qubit"):
        fq.twin.prepare_probe_bank(
            target="quafu:Shenglian",
            physical_qubits=(20, 27, 34),
            probes=_identifiable_probes(),
            parameter_family="local_pauli_readout",
        )

    with pytest.raises(ValueError, match="parameter_family"):
        fq.twin.prepare_probe_bank(
            target="quafu:Shenglian",
            physical_qubits=(20, 27),
            probes=_identifiable_probes(),
            parameter_family="arbitrary_kraus",
        )

    duplicate = (_identifiable_probes()[0], _identifiable_probes()[0])
    with pytest.raises(ValueError, match="unique identities"):
        _bank(duplicate)


def test_nonlocal_dynamic_and_explicit_measurement_probes_fail_closed() -> None:
    three_qubit = fq.Circuit(3).gate(
        "custom_three",
        (0, 1, 2),
        matrix=[[1 if row == column else 0 for column in range(8)] for row in range(8)],
    )
    with pytest.raises(ValueError, match="one- and two-qubit"):
        fq.twin.prepare_probe_bank(
            target="acme:qpu",
            physical_qubits=(1, 2, 3),
            probes=(three_qubit,),
            parameter_family="local_pauli_readout",
        )

    measured = CircuitIR(
        n_wires=2,
        instructions=fq.Circuit(2).x(0).to_ir().instructions,
        measurements=(MeasurementNode("sample", (0, 1), shots=100),),
    )
    with pytest.raises(ValueError, match="implicit computational-basis"):
        _bank((measured,))


def test_constructor_recomputes_rank_and_identifiable_parameters() -> None:
    bank = _bank()

    with pytest.raises(ValueError, match="matrix_rank"):
        replace(bank, matrix_rank=bank.matrix_rank - 1)
    with pytest.raises(ValueError, match="unidentified_parameters"):
        replace(bank, unidentified_parameters=(bank.parameter_names[0],))
    with pytest.raises(ValueError, match="mapping and couplers"):
        replace(bank, parameter_names=tuple(reversed(bank.parameter_names)))
    tampered_supports = (
        (bank.probe_parameter_names[0][0], (bank.parameter_names[0],)),
        *bank.probe_parameter_names[1:],
    )
    with pytest.raises(ValueError, match="frozen design matrix"):
        replace(bank, probe_parameter_names=tampered_supports)
    with pytest.raises(ValueError, match="bound to each probe"):
        replace(
            bank,
            design_row_labels=(
                f"{'0' * 64}:z[Q20]",
                *bank.design_row_labels[1:],
            ),
        )
    with pytest.raises(ValueError, match="ideal row probabilities"):
        replace(
            bank,
            ideal_row_probabilities=((0.4, 0.4), *bank.ideal_row_probabilities[1:]),
        )


def test_probe_bank_is_immutable_and_has_no_execution_or_estimation_authority() -> None:
    bank = _bank()

    with pytest.raises(FrozenInstanceError):
        bank.matrix_rank = 0  # type: ignore[misc]
    for name in (
        "submit",
        "poll",
        "retry",
        "cancel",
        "estimate",
        "fit",
        "promote",
    ):
        assert not hasattr(bank, name)
