"""Compose connected Twin cells and inspect one circuit mapping offline.

Each Twin/support pair must come from a prospectively validated cell on the
same provider, backend, and calibration capture. Adjust the artifact paths and
physical mapping to the cells produced by your validation workflow.
"""

import flagquantum as fq

# Loading and composition are offline. No provider credentials are required.
twin_a = fq.twin.load_twin("cell-a-twin.json")
support_a = fq.twin.load_circuit_support("cell-a-support.json")
twin_b = fq.twin.load_twin("cell-b-twin.json")
support_b = fq.twin.load_circuit_support("cell-b-support.json")

region = fq.twin.compose_connected_region([(twin_a, support_a), (twin_b, support_b)])

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
coverage = region.coverage_report(
    circuit,
    physical_qubits=(20, 27, 34),
)

print(coverage.status)
print(coverage.covered_qubits)
print(coverage.covered_directed_couplers)
print(coverage.missing_qubits)
print(coverage.missing_directed_couplers)

# Local cell error bounds are deliberately not composed into region accuracy.
assert coverage.tv_error_bound is None
assert coverage.confidence_level is None
