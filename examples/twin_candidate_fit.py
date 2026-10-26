"""Build and persist one fitted QPU Twin candidate entirely offline.

The deterministic counts are synthetic API examples, not QPU evidence. In a
hardware workflow, ``hardware_reports`` comes from distinct tasks already
validated by ``TwinExperiment.validate_result(...)``.
"""

from __future__ import annotations

import hashlib

import flagquantum as fq
from flagquantum.noise import (
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
)

SHOTS = 4096
PARAMETER_COUNTS = (64, 32, 96, 128, 48)


def probes() -> tuple[fq.Circuit, ...]:
    return (
        fq.Circuit(2),
        fq.Circuit(2).x(0),
        fq.Circuit(2).x(1),
        fq.Circuit(2).cx(0, 1),
    )


def incumbent_twin() -> fq.twin.QPUDigitalTwin:
    profile = DeviceNoiseProfile(
        qubits=(
            QubitNoiseCalibration(0, t1=40_000.0, t2=60_000.0),
            QubitNoiseCalibration(1, t1=35_000.0, t2=50_000.0),
        ),
        gate_durations=(
            GateDuration("h", 64.0, wires=(0,)),
            GateDuration("x", 64.0, wires=(0,)),
            GateDuration("h", 64.0, wires=(1,)),
            GateDuration("x", 64.0, wires=(1,)),
            GateDuration("cx", 224.0, wires=(0, 1)),
        ),
        source="offline-candidate-example",
        captured_at="2026-08-14T10:30:00+08:00",
        time_unit="ns",
    )
    return fq.twin.from_noise_model(
        NoiseModel.from_device_profile(profile),
        target="quafu:Shenglian",
        qubits=(20, 27),
    )


def synthetic_counts(circuit: fq.Circuit) -> dict[str, int]:
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
    twin: fq.twin.QPUDigitalTwin,
    circuit: fq.Circuit,
    task_index: int,
) -> fq.twin.TwinHardwareReport:
    prediction = twin.predict(circuit)
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
    incumbent = incumbent_twin()
    probe_circuits = probes()
    bank = fq.twin.prepare_probe_bank(
        target="quafu:Shenglian",
        physical_qubits=(20, 27),
        probes=probe_circuits,
        parameter_family="local_pauli_readout",
    )
    reports = tuple(
        synthetic_report(incumbent, circuit, index)
        for index, circuit in enumerate(probe_circuits, start=1)
    )
    estimate = fq.twin.estimate_probe_parameters(bank, reports)
    candidate = fq.twin.fit_candidate(
        incumbent,
        estimate,
        captured_at="2026-08-15T10:30:00+08:00",
    )

    destination = "twin-candidate.json"
    fq.twin.dump_twin(candidate, destination)
    restored = fq.twin.load_twin(destination)
    holdout = fq.Circuit(2).h(0).cx(0, 1)
    prediction = restored.predict(holdout)

    print("incumbent:", incumbent.snapshot.identity)
    print("candidate:", restored.snapshot.identity)
    print("estimate:", candidate.fit.parameter_estimate_identity)
    print("probe bank:", candidate.fit.probe_bank_identity)
    print("fitted:", candidate.fit.fitted_parameters)
    print("retained:", candidate.fit.retained_parameters)
    print("holdout TV from ideal:", prediction.total_variation_from_ideal)


if __name__ == "__main__":
    main()
