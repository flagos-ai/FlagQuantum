"""Run a complete offline Twin parameter-estimation example.

The counts below are deterministic synthetic observations for API education and
tests. They are not QPU evidence. A hardware workflow supplies distinct reports
returned by ``TwinExperiment.validate_result(...)`` instead.
"""

from __future__ import annotations

import hashlib

import flagquantum as fq

SHOTS = 4096
SNAPSHOT_IDENTITY = "a" * 64
PARAMETER_COUNTS = (64, 32, 96, 128, 48)


def probes() -> tuple[fq.Circuit, ...]:
    """Return a small full-rank local probe design."""

    return (
        fq.Circuit(2),
        fq.Circuit(2).x(0),
        fq.Circuit(2).x(1),
        fq.Circuit(2).cx(0, 1),
    )


def synthetic_counts(circuit: fq.Circuit) -> dict[str, int]:
    """Encode a known local model as exact finite counts."""

    readout_0, readout_1, single_0, single_1, two_qubit = PARAMETER_COUNTS
    identities = tuple(probe.to_ir().content_hash for probe in probes())
    identity = circuit.to_ir().content_hash
    counts = [0, 0, 0, 0]
    if identity == identities[0]:
        counts[1], counts[2] = readout_1, readout_0
        counts[0] = SHOTS - sum(counts)
    elif identity == identities[1]:
        counts[0], counts[3] = readout_0 + single_0, readout_1
        counts[2] = SHOTS - sum(counts)
    elif identity == identities[2]:
        counts[0], counts[3] = readout_1 + single_1, readout_0
        counts[1] = SHOTS - sum(counts)
    else:
        counts[1], counts[2], counts[3] = readout_1, readout_0, two_qubit
        counts[0] = SHOTS - sum(counts)
    return {format(index, "02b"): count for index, count in enumerate(counts) if count}


def synthetic_report(
    circuit: fq.Circuit, task_index: int
) -> fq.twin.TwinHardwareReport:
    """Build one deterministic report with the same public result contract."""

    ideal = tuple(float(value) for value in circuit.probabilities().reshape(-1))
    prediction = fq.twin.TwinPrediction(
        snapshot_identity=SNAPSHOT_IDENTITY,
        circuit_identity=circuit.to_ir().content_hash,
        n_wires=2,
        ideal_probabilities=ideal,
        twin_probabilities=ideal,
        total_variation_from_ideal=0.0,
    )
    counts = synthetic_counts(circuit)
    program_identity = hashlib.sha256(f"program-{task_index}".encode()).hexdigest()
    return fq.twin.TwinHardwareReport(
        experiment_identity=hashlib.sha256(
            f"experiment-{task_index}".encode()
        ).hexdigest(),
        provider="quafu",
        backend_name="Shenglian",
        task_id=f"synthetic-task-{task_index}",
        submitted_qasm_identity=program_identity,
        executed_qasm_identity=program_identity,
        counts=counts,
        validation=prediction.compare_counts(counts),
        result_metadata={"target_qubits": [20, 27]},
    )


def main() -> None:
    probe_circuits = probes()
    bank = fq.twin.prepare_probe_bank(
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        probes=probe_circuits,
        parameter_family="local_pauli_readout",
    )
    hardware_reports = tuple(
        synthetic_report(circuit, index)
        for index, circuit in enumerate(probe_circuits, start=1)
    )
    estimate = fq.twin.estimate_probe_parameters(
        bank,
        hardware_reports,
        confidence_level=0.95,
    )

    print(estimate.parameter_values)
    print(estimate.confidence_intervals)
    print(estimate.identifiability_status)
    print(estimate.goodness_of_fit)
    print(estimate.boundary_parameters)


if __name__ == "__main__":
    main()
