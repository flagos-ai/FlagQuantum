"""Compose validated Twin cells into one offline regional prediction model.

The input artifacts must come from compatible cells on the same target and
calibration capture. Adjust the paths and physical mapping to your artifacts.
"""

import flagquantum as fq

twin_a = fq.twin.load_twin("cell-a-twin.json")
support_a = fq.twin.load_circuit_support("cell-a-support.json")
twin_b = fq.twin.load_twin("cell-b-twin.json")
support_b = fq.twin.load_circuit_support("cell-b-support.json")

region_twin = fq.twin.compose_region_twin([(twin_a, support_a), (twin_b, support_b)])

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
prediction = region_twin.predict(
    circuit,
    physical_qubits=(20, 27, 34),
)

print(region_twin.target)
print(region_twin.physical_qubits)
print(prediction.twin_probabilities)
print(prediction.total_variation_from_ideal)

# This is model output, not a region-level hardware accuracy statement.
assert not hasattr(region_twin, "evidence_report")
