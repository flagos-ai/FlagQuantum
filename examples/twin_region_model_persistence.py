"""Persist one composed regional Twin and restore it without its source cells.

This example builds a provider-neutral regional model offline, stores the
composed model once, and reloads it. The restored model is exactly the object
an application passes to a released regional Twin; see
``examples/twin_region_release_assessment.py`` for the assessment step, which
accepts only a model whose identity, snapshot, target, and ordered mapping match
the release.
"""

import flagquantum as fq
from flagquantum.noise import (
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
)

MAPPING = (20, 27, 34)


def _cell(qubits: tuple[int, int]) -> fq.twin.QPUDigitalTwin:
    profile = DeviceNoiseProfile(
        qubits=tuple(
            QubitNoiseCalibration(wire, t1=40_000.0, t2=60_000.0)
            for wire in range(len(qubits))
        ),
        gate_durations=(GateDuration("h", 64.0), GateDuration("cx", 224.0)),
        source="example-regional-calibration",
        captured_at="2026-08-15T10:30:00+08:00",
    )
    return fq.twin.from_noise_model(
        NoiseModel.from_device_profile(profile),
        target="acme:regional-qpu",
        qubits=qubits,
    )


def _support(twin: fq.twin.QPUDigitalTwin) -> fq.twin.TwinCircuitSupport:
    qubits = twin.snapshot.physical_qubits
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    evidence = fq.twin.TwinEvidenceEnvelope(
        snapshot_identity=twin.snapshot.identity,
        physical_qubits=qubits,
        supported_operations=("h", "cx"),
        maximum_instruction_count=8,
        verified_circuit_identities=(circuit.to_ir().content_hash,),
        evidence_identity=(f"{qubits[0]:02x}{qubits[1]:02x}" * 16)[:64],
        verified_tv_error_bound=0.04,
        estimated_tv_error_bound=0.08,
        confidence_level=0.95,
    )
    return fq.twin.TwinCircuitSupport(
        evidence=evidence,
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=5,
    )


cell_a, cell_b = _cell((20, 27)), _cell((27, 34))
region_twin = fq.twin.compose_region_twin(
    ((cell_a, _support(cell_a)), (cell_b, _support(cell_b)))
)

# Release artifacts are written once; the composed model now travels with them.
fq.twin.dump_region_twin(region_twin, "region-twin.json")
restored = fq.twin.load_region_twin("region-twin.json")

assert restored.identity == region_twin.identity
assert restored.target == region_twin.target
assert restored.physical_qubits == MAPPING

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
before = region_twin.predict(circuit, physical_qubits=MAPPING)
after = restored.predict(circuit, physical_qubits=MAPPING)
assert after == before
assert after.twin_probabilities == before.twin_probabilities

print(restored.identity)
print(restored.region.directed_couplers)
print(after.twin_probabilities)
print(after.total_variation_from_ideal)

# The comment markers below show the released-model workflow this artifact
# exists for; it needs a release manifest produced on the same regional model.
#
# release = fq.twin.load_region_release("region-release.json")
# assessment = release.assess(restored, circuit, physical_qubits=MAPPING)
# print(assessment.status)
# print(assessment.prediction)
# print(assessment.reasons)
