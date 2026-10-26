"""Scenarios for bounded estimation from frozen Twin probe banks."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

import flagquantum as fq

pytestmark = pytest.mark.integration

_SHOTS = 4096
_SNAPSHOT = "a" * 64


def _probes():
    return (
        fq.Circuit(2),
        fq.Circuit(2).x(0),
        fq.Circuit(2).x(1),
        fq.Circuit(2).cx(0, 1),
    )


def _bank(probes=None):
    return fq.twin.prepare_probe_bank(
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        probes=_probes() if probes is None else probes,
        parameter_family="local_pauli_readout",
    )


def _counts_for(circuit, values):
    readout_0, readout_1, single_0, single_1, two_qubit = values
    identity = circuit.to_ir().content_hash
    identities = tuple(probe.to_ir().content_hash for probe in _probes())
    counts = [0, 0, 0, 0]
    if identity == identities[0]:
        counts[1] = readout_1
        counts[2] = readout_0
        counts[0] = _SHOTS - sum(counts)
    elif identity == identities[1]:
        counts[0] = readout_0 + single_0
        counts[3] = readout_1
        counts[2] = _SHOTS - sum(counts)
    elif identity == identities[2]:
        counts[0] = readout_1 + single_1
        counts[3] = readout_0
        counts[1] = _SHOTS - sum(counts)
    else:
        counts[1] = readout_1
        counts[2] = readout_0
        counts[3] = two_qubit
        counts[0] = _SHOTS - sum(counts)
    return {format(index, "02b"): count for index, count in enumerate(counts) if count}


def _report(circuit, values, *, task_index, snapshot=_SNAPSHOT, mapping=(20, 27)):
    ideal = tuple(float(value) for value in circuit.probabilities().reshape(-1))
    prediction = fq.twin.TwinPrediction(
        snapshot_identity=snapshot,
        circuit_identity=circuit.to_ir().content_hash,
        n_wires=2,
        ideal_probabilities=ideal,
        twin_probabilities=ideal,
        total_variation_from_ideal=0.0,
    )
    counts = _counts_for(circuit, values)
    executed_identity = hashlib.sha256(f"executed-{task_index}".encode()).hexdigest()
    return fq.twin.TwinHardwareReport(
        experiment_identity=hashlib.sha256(
            f"experiment-{task_index}".encode()
        ).hexdigest(),
        provider="quafu",
        backend_name="Shenglian",
        task_id=f"task-{task_index}",
        submitted_qasm_identity=executed_identity,
        executed_qasm_identity=executed_identity,
        counts=counts,
        validation=prediction.compare_counts(counts),
        result_metadata={"target_qubits": list(mapping)},
    )


def _reports(values=(64, 32, 96, 128, 48)):
    return tuple(
        _report(probe, values, task_index=index)
        for index, probe in enumerate(_probes(), start=1)
    )


def test_identifiable_bank_recovers_known_bounded_parameters() -> None:
    bank = _bank()
    expected_counts = (64, 32, 96, 128, 48)

    estimate = fq.twin.estimate_probe_parameters(
        bank,
        _reports(expected_counts),
        confidence_level=0.95,
    )

    expected = tuple(value / _SHOTS for value in expected_counts)
    assert estimate.identifiability_status == "identifiable"
    assert tuple(value for _, value in estimate.parameter_values) == pytest.approx(
        expected
    )
    assert tuple(name for name, _ in estimate.parameter_values) == bank.parameter_names
    assert estimate.unidentified_parameters == ()
    assert estimate.residual_root_mean_square == pytest.approx(0.0, abs=1e-12)
    assert estimate.maximum_absolute_residual == pytest.approx(0.0, abs=1e-12)
    assert estimate.goodness_of_fit == pytest.approx(1.0)
    assert estimate.probe_bank_identity == bank.identity
    assert estimate.target == "quafu:Shenglian"
    assert estimate.physical_qubits == (20, 27)
    assert len(estimate.identity) == 64
    assert all(
        lower <= value <= upper
        for (_, value), (_, lower, upper) in zip(
            estimate.parameter_values, estimate.confidence_intervals, strict=True
        )
    )


def test_report_order_and_repeated_observations_are_deterministic() -> None:
    bank = _bank()
    first = _reports()
    repeated = tuple(
        replace(report, task_id=f"repeat-{index}")
        for index, report in enumerate(first, start=1)
    )

    left = fq.twin.estimate_probe_parameters(bank, (*first, *repeated))
    right = fq.twin.estimate_probe_parameters(
        bank, tuple(reversed((*first, *repeated)))
    )

    assert left == right
    assert left.identity == right.identity
    assert all(shots == 2 * _SHOTS for _, shots in left.shots_by_probe)


def test_underdetermined_bank_fails_closed_and_names_missing_parameters() -> None:
    bank = _bank((fq.Circuit(2), fq.Circuit(2).x(0)))

    assert bank.identifiability_status == "underdetermined"
    with pytest.raises(ValueError, match="unidentified") as error:
        fq.twin.estimate_probe_parameters(bank, _reports())
    assert bank.unidentified_parameters[0] in str(error.value)


def test_reused_invalid_and_mismatched_reports_fail_closed() -> None:
    bank = _bank()
    reports = _reports()

    with pytest.raises(ValueError, match="distinct and cannot be reused"):
        fq.twin.estimate_probe_parameters(bank, (*reports, reports[0]))
    with pytest.raises(ValueError, match="distinct provider tasks"):
        fq.twin.estimate_probe_parameters(
            bank, (*reports[:-1], replace(reports[-1], task_id=reports[0].task_id))
        )
    with pytest.raises(ValueError, match="physical mapping"):
        fq.twin.estimate_probe_parameters(
            bank,
            (
                *reports[:-1],
                replace(reports[-1], result_metadata={"target_qubits": [27, 20]}),
            ),
        )
    with pytest.raises(ValueError, match="one frozen Twin snapshot"):
        fq.twin.estimate_probe_parameters(
            bank,
            (
                *reports[:-1],
                replace(
                    reports[-1],
                    validation=replace(
                        reports[-1].validation, snapshot_identity="b" * 64
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="match report counts"):
        fq.twin.estimate_probe_parameters(
            bank, (*reports[:-1], replace(reports[-1], counts={"00": _SHOTS}))
        )
    with pytest.raises(ValueError, match="missing probe circuits"):
        fq.twin.estimate_probe_parameters(bank, reports[:-1])


def test_boundary_parameters_are_explicit_and_result_has_no_mutation_authority() -> (
    None
):
    bank = _bank()
    estimate = fq.twin.estimate_probe_parameters(
        bank,
        _reports((0, 0, 0, 0, 0)),
    )

    assert estimate.boundary_parameters == bank.parameter_names
    assert all(value == pytest.approx(0.0) for _, value in estimate.parameter_values)
    with pytest.raises(FrozenInstanceError):
        estimate.goodness_of_fit = 0.0  # type: ignore[misc]
    for name in ("submit", "poll", "retry", "fit_candidate", "promote", "route"):
        assert not hasattr(estimate, name)


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
def test_invalid_confidence_fails_closed(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        fq.twin.estimate_probe_parameters(
            _bank(), _reports(), confidence_level=confidence
        )
