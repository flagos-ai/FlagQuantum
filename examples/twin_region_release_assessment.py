"""Ask whether one exact circuit is covered by a regional Twin release.

Run this only after a release manifest exists. Reconstruct the released
regional model from the same cell artifacts that produced the release; an
assessment against any other model, snapshot, target, or mapping fails closed.
"""

import flagquantum as fq

release = fq.twin.load_region_release("region-release.json")
region_twin = fq.twin.compose_region_twin(
    [
        (
            fq.twin.load_twin("cell-a-twin.json"),
            fq.twin.load_circuit_support("cell-a-support.json"),
        ),
        (
            fq.twin.load_twin("cell-b-twin.json"),
            fq.twin.load_circuit_support("cell-b-support.json"),
        ),
    ]
)

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
assessment = release.assess(region_twin, circuit, physical_qubits=(20, 27, 34))

print(assessment.status)
print(assessment.prediction)
print(assessment.reasons)
print(assessment.release_identity)

assert assessment.release_identity == release.identity
if assessment.status == "released_exact_circuit":
    assert assessment.reasons == ()
    assert assessment.prediction is not None
else:
    assert assessment.prediction is None
    assert assessment.reasons

# A release states no per-circuit confidence level and authorizes no routing.
assert release.scope == "exact_circuits"
assert release.routing_authorized is False
